"""Generates a clip with its audio with LTX-2.5, its transformer split over NeuronCores, or all on CPU.

The distilled pipeline of models/LTX-2.5-Diffusers generates in one stage, with the settings of its model card. With
--two-stage, it follows the two-stage recipe of the model card: it generates the clip at --width x --height as
latents, the latent upsampler upscales them x2 on CPU, and the pipeline denoises them again at twice the size. On
Neuron, `parallel` splits the transformer, and the decoder of the video VAE by frames, over the first --tp NeuronCores
of NEURON_RT_VISIBLE_CORES, one torch.distributed rank per core, and torch.compile(backend="neuron") compiles the
transformer's blocks and the decoder. With --sp, --sp groups of --tp cores split the video tokens of the transformer,
and the decoder runs on all of their cores. Only rank 0 runs the pipeline, its other models on CPU. It sends the
arguments of each transformer call and of the decode to the other ranks, which run their slices of the transformer on
them and decode their frames, which rank 0 gathers. With --cpu, the whole pipeline runs on CPU.

The clip goes to outputs/NAME.mp4, 4 of its frames to outputs/NAME.png and, with --npz, its frames and audio to
outputs/NAME.npz.
NAME is video-gen-demo-WIDTHxHEIGHT by default, after the size of the clip, with -cpu on CPU. With several prompts,
the process makes one clip per prompt, one after the other, and their names end with -1, -2...
outputs/ is in the repository, and models/ in the working directory of the demo, `launch.WORK_DIR`.

With --film, the process makes the shots of a film, NAME-shot-1, NAME-shot-2..., then the film, NAME-full, its shots
one after the other. A shot can start from the last frames of an earlier shot, which the condition pipeline of diffusers
keeps as its first frames. From the last frame of a shot, it returns to its subject after a cut. From its last second,
it continues its motion. A shot can also end on the first frames of a later shot, which the condition pipeline takes as
keyframes: it then leads into that shot, which the process makes first. The film leaves out the frames a shot starts
from, which repeat the earlier shot, and those of a shot from its keyframes on, which the later shot repeats. NAME is
FILM-WIDTHxHEIGHT by default, after the file of the film.

Run with:
    ./run.sh generate.py [--cpu] [--prompt PROMPT... | --film FILM] [--seed SEED] [--name NAME] [--width WIDTH]
        [--height HEIGHT] [--frames FRAMES] [--two-stage] [--tp TP] [--sp SP] [--npz]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import launch

MODEL_DIR = os.path.join(launch.WORK_DIR, "models", "LTX-2.5-Diffusers")
# Unlike the weights, the clips are small: they stay in the repository, where the editor shows them.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# On 2 NeuronCores, the transformer ran out of memory at 960x544 for 10 s.
TP = 4
RANK = int(os.environ.get("RANK", "0"))
# By default, 10 seconds at 1024x512. LTX-2.5 takes one frame more than a multiple of 8. A frame of 1024x512 has 512
# video tokens: their count stays a multiple of 128, so torch-neuronx runs the video self-attention with its NKI flash
# attention kernel.
WIDTH, HEIGHT, NUM_FRAMES, FPS = 1024, 512, 241, 24.0
PROMPT = (
    "A red fox trots through fresh snow in a sunlit pine forest, its breath steaming in the cold air. The camera "
    "tracks alongside it at ground level. Snow crunches under its paws and birds sing in the distance."
)


def log(msg: str) -> None:
    if RANK == 0:
        print(f"[generate] {msg}", flush=True)


def on_neuron(transformer, sp_group=None):
    """The forward of `transformer`, its blocks compiled for the NeuronCore of this rank: it takes and returns tensors
    on the host. With `sp_group`, the sequence-parallel group of this rank, it runs on the video tokens of this rank,
    and returns the video output of all of them. A timestep per video token, as for a clip that starts from frames, it
    runs with two timesteps (see `timesteps`)."""
    import contextlib

    import torch
    from torch.utils._pytree import tree_map

    import parallel
    import timesteps

    device = f"neuron:{torch.neuron.current_device()}"
    # The blocks share one compiled graph, so a new process traces and lowers a single block. The rest of the forward
    # runs op by op.
    transformer.compile_repeated_blocks(backend="neuron", fullgraph=True, dynamic=False)
    forward = transformer.forward

    def call(*args, **kwargs):
        if sp_group is not None:
            kwargs = parallel.split_video_tokens(kwargs, sp_group)
        args, kwargs = tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, (args, kwargs))
        per_token = kwargs["timestep"].ndim > 1
        with timesteps.two_timesteps(transformer) if per_token else contextlib.nullcontext():
            output = forward(*args, **kwargs)
        if sp_group is not None:
            output = (parallel.gather_video_tokens(output[0], sp_group), *output[1:])
        return tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, output)

    return call


def decode_on_neuron(vae):
    """The decode of the video VAE `vae`, its decoder compiled for the NeuronCore of this rank: it takes latents on the
    host and returns the frames it decodes on the host."""
    import torch

    device = f"neuron:{torch.neuron.current_device()}"
    # Split by frames, the decoder exchanges frames with the other ranks between its compiled graphs. dynamo compiles
    # the convolution after each exchange once per shape of its input, for more shapes than its default limit of 8,
    # past which it runs the convolutions op by op.
    torch._dynamo.config.recompile_limit = 64
    vae.decoder.compile(backend="neuron", fullgraph=False, dynamic=False)
    decode = vae.decode

    def call(latents, *args, **kwargs):
        with torch.no_grad():
            return decode(latents.to(device), *args, **kwargs)[0].cpu()

    return call


def split_decode(vae, group) -> None:
    """Makes the video VAE `vae`, split by frames over the NeuronCores of the ranks of the gloo group `group`, decode its
    frames on the core of this rank, after it sends the arguments of the decode to the other ranks, which decode theirs
    (`serve`), then gather them. Its decode returns a tuple, as with `return_dict=False`."""
    import torch
    import torch.distributed as dist

    decode = decode_on_neuron(vae)

    def call(*args, **kwargs):
        dist.broadcast_object_list([("decode", args, kwargs)], src=0, group=group)
        frames = [None] * dist.get_world_size(group)
        dist.gather_object(decode(*args, **kwargs), frames, dst=0, group=group)
        return (torch.cat(frames, dim=2),)

    vae.decode = call


def timed(transformer, calls: list, group=None, sp_group=None) -> None:
    """Makes `transformer` append the duration of each of its calls to `calls`.

    With `group`, the gloo group of the ranks that split `transformer` over their NeuronCores, it runs on the core of
    this rank, after it sends the arguments of the call to the other ranks, which run their slices on them (`serve`).
    `sp_group` is the sequence-parallel group of this rank, if any (see `on_neuron`).
    """
    import torch.distributed as dist

    forward = transformer.forward if group is None else on_neuron(transformer, sp_group)

    def call(*args, **kwargs):
        t0 = time.time()
        if group is not None:
            dist.broadcast_object_list([("transformer", args, kwargs)], src=0, group=group)
        output = forward(*args, **kwargs)
        calls.append(time.time() - t0)
        return output

    transformer.forward = call


def serve(transformer, vae, group, sp_group=None) -> None:
    """Runs on the NeuronCore of this rank, on the arguments that rank 0 sends over `group`, the slice of `transformer`
    of this rank for each transformer call (see `timed`), and the decoder of `vae` on the frames of this rank for each
    decode, which it then sends to rank 0 (see `split_decode`), until it sends None. `sp_group` is the
    sequence-parallel group of this rank, if any (see `on_neuron`)."""
    import torch
    import torch.distributed as dist

    forward = on_neuron(transformer, sp_group)
    decode = decode_on_neuron(vae)
    while True:
        message = [None]
        dist.broadcast_object_list(message, src=0, group=group)
        if message[0] is None:
            return
        kind, args, kwargs = message[0]
        if kind == "decode":
            dist.gather_object(decode(*args, **kwargs), None, dst=0, group=group)
            continue
        # Without grad, as the pipeline runs on rank 0: with grad, this rank would compile another graph, which keeps
        # the activations for a backward pass.
        with torch.no_grad():
            forward(*args, **kwargs)


def pack_parameters(module, alignment: int = 4096, max_buffer: int = 2**30) -> None:
    """Moves the parameters of `module`, on a NeuronCore, through the host into views of buffers of `max_buffer` bytes.

    torch-neuronx's allocator binds device memory lazily, and as it binds the parameters of a model one by one, it leaves
    most of the room of the segments it creates for them unused: for the 9.45 GiB of the transformer's slice, it holds
    22.96 GiB, which `empty_cache` cannot release while the parameters live. In buffers, they take only their size. With
    one buffer of 9.45 GiB, the runtime fails to load the graphs that have collectives.
    """
    import gc

    import torch
    from torch.distributed.tensor import DTensor

    params = []
    for name, param in list(module.named_parameters()):
        local = param.to_local() if isinstance(param, DTensor) else param.data
        layout = (param.device_mesh, param.placements) if isinstance(param, DTensor) else None
        params.append((name, local.cpu(), layout, param.requires_grad, local.device))
        path, _, attr = name.rpartition(".")
        module.get_submodule(path).register_parameter(attr, None)
    # Freed before the buffers take their place, the parameters never take room twice on the device.
    gc.collect()
    torch.neuron.synchronize()
    torch.neuron.empty_cache()
    # The buffer of each parameter, and its offset in it.
    places, sizes = [], [0]
    for _, host, *_ in params:
        size = -(-host.nbytes // alignment) * alignment
        if sizes[-1] and sizes[-1] + size > max_buffer:
            sizes.append(0)
        places.append((len(sizes) - 1, sizes[-1]))
        sizes[-1] += size
    buffers = [torch.empty(size, dtype=torch.uint8, device=params[0][4]) for size in sizes]
    for (name, host, layout, requires_grad, _), (index, offset) in zip(params, places):
        view = buffers[index][offset : offset + host.nbytes].view(host.dtype).view(host.shape)
        view.copy_(host)
        if layout is not None:
            view = DTensor.from_local(view, *layout, run_check=False)
        path, _, attr = name.rpartition(".")
        module.get_submodule(path).register_parameter(attr, torch.nn.Parameter(view, requires_grad=requires_grad))


def load_models(sp: int = 1):
    """The slice of the LTX-2.5 transformer of this rank, and the video VAE, its decoder on the NeuronCore of this rank,
    both split over the NeuronCores of all ranks, a gloo group of all ranks, which sends the arguments of their calls,
    and the sequence-parallel group of this rank, or None.

    With `sp` above 1, `sp` groups of ranks, each on its own cores, split the video tokens of the transformer, and the
    ranks of each group split its weights.
    """
    import torch
    import torch.distributed as dist
    from diffusers import AutoencoderKLLTX2Video
    from torch.distributed.device_mesh import DeviceMesh

    import parallel

    dist.init_process_group(backend="neuron")
    world_size = dist.get_world_size()
    if sp > 1:
        # Consecutive ranks share a chip, and so do the ranks that split the weights.
        mesh = DeviceMesh("neuron", torch.arange(world_size).reshape(sp, world_size // sp), mesh_dim_names=("sp", "tp"))
        tp_mesh, sp_group = mesh["tp"], mesh["sp"].get_group()
    else:
        tp_mesh, sp_group = DeviceMesh("neuron", list(range(world_size))), None
    transformer = parallel.load_transformer(os.path.join(MODEL_DIR, "transformer"), tp_mesh, sp_group)
    pack_parameters(transformer)
    vae = AutoencoderKLLTX2Video.from_pretrained(os.path.join(MODEL_DIR, "vae"), dtype=torch.bfloat16)
    # The pipeline normalizes the latents on the host with the other tensors of the VAE.
    vae.decoder.to(f"neuron:{torch.neuron.current_device()}")
    pack_parameters(vae.decoder)
    parallel.split_by_frames(vae, dist.group.WORLD)
    return transformer, vae, dist.new_group(backend="gloo"), sp_group


def on_host(pipeline_class):
    """The diffusers pipeline class `pipeline_class`, made to run on the host."""
    import torch

    class HostPipeline(pipeline_class):
        # diffusers runs a pipeline on the device of its accelerated models. All
        # models but the transformer run on the host, and the transformer moves
        # its own inputs.
        @property
        def _execution_device(self):
            return torch.device("cpu")

    return HostPipeline


def load_pipeline(calls: list, transformer=None, vae=None, group=None, sp_group=None):
    """The distilled LTX-2.5 pipeline in bf16, its tensors on the host. Its transformer and video VAE run on CPU, or
    are `transformer` and `vae` on Neuron, with `group` and `sp_group`, as `load_models` returns them."""
    import torch
    from diffusers import LTX2Pipeline

    components = {} if transformer is None else {"transformer": transformer, "vae": vae}
    # The processor and the prompt enhancer serve prompt enhancement only.
    pipe = on_host(LTX2Pipeline).from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, processor=None, prompt_enhancer=None, **components
    )
    timed(pipe.transformer, calls, group, sp_group)
    if group is not None:
        split_decode(pipe.vae, group)
    return pipe


def load_upsampler(pipe):
    """The x2 latent upsampler of LTX-2.5 in bf16, on the host, as a pipeline that upscales the latents of `pipe`."""
    import torch
    from diffusers import LTX2LatentUpsamplePipeline, LTX2LatentUpsamplerModel

    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        os.path.join(MODEL_DIR, "latent_upsampler"), dtype=torch.bfloat16
    )
    # On latents, the pipeline reads only the compression ratios of the VAE.
    return on_host(LTX2LatentUpsamplePipeline)(vae=pipe.vae, latent_upsampler=upsampler)


def load_condition_pipeline(pipe):
    """The LTX-2.5 pipeline that starts a clip from frames, on the models of `pipe`. Its VAE encodes the frames on the
    host."""
    from diffusers import LTX2ConditionPipeline

    # `from_pipe` would cast the models to its dtype, fp32 by default.
    return on_host(LTX2ConditionPipeline)(**pipe.components)


def keyframe_index(num_frames: int, count: int) -> int:
    """The latent index of the `count` frames, one more than a multiple of 8, that a clip of `num_frames` frames ends
    on. As the VAE encodes the frames after the first by 8, the condition pipeline places them from frame
    (index - 1) * 8 + 1 of the clip, and the last of them on the first frame of its last latent."""
    return (num_frames - 1) // 8 - (count - 1) // 8


def generate(
    pipe, prompt: str, seed: int, width: int, height: int, num_frames: int, upsample_pipe=None, start=None, end=None
):
    """The video, as a tensor of frames in [0, 1], (batch, frames, channels, height, width), and the audio of the clip
    of `prompt`.

    With `upsample_pipe`, the clip follows the two-stage distilled recipe of the model card: `pipe` makes it at `width`
    x `height` as latents, `upsample_pipe` upscales them x2, and `pipe` denoises them again at twice the size, over
    the last sigmas of the schedule.

    With `start`, frames in uint8 RGB, the clip starts from them: `pipe`, a pipeline of `load_condition_pipeline`, keeps
    them as its first frames, which it scales to the size of each stage. With `end`, frames in uint8 RGB, the clip ends
    on them: `pipe` takes them as keyframes (see `keyframe_index`), after which the clip goes on for 7 frames.
    """
    import torch
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )

    # Otherwise each stage runs the text encoder again, on CPU.
    prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
        prompt=prompt, do_classifier_free_guidance=False, device=torch.device("cpu")
    )
    settings = dict(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_frames=num_frames,
        frame_rate=FPS,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        # Drawn on the host, the noise is the same on Neuron and on CPU.
        generator=torch.Generator("cpu").manual_seed(seed),
        return_dict=False,
    )
    if start is not None or end is not None:
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        settings["conditions"] = []
        if start is not None:
            settings["conditions"].append(LTX2VideoCondition(frames=start, index=0))
        if end is not None:
            settings["conditions"].append(LTX2VideoCondition(frames=end, index=keyframe_index(num_frames, len(end))))
    if upsample_pipe is None:
        return pipe(width=width, height=height, sigmas=DISTILLED_SIGMA_VALUES, output_type="pt", **settings)
    t0 = time.time()
    latents, audio_latents = pipe(
        width=width, height=height, sigmas=DISTILLED_SIGMA_VALUES, output_type="latent", **settings
    )
    log(f"stage 1: {time.time() - t0:.0f} s")
    t0 = time.time()
    with torch.no_grad():
        latents = upsample_pipe(latents=latents, output_type="latent", return_dict=False)[0]
    log(f"upsampling: {time.time() - t0:.0f} s")
    t0 = time.time()
    video, audio = pipe(
        latents=latents,
        audio_latents=audio_latents,
        # The plain pipeline takes the size of the latents, the condition pipeline sizes its frames by these.
        width=2 * width,
        height=2 * height,
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        output_type="pt",
        **settings,
    )
    log(f"stage 2: {time.time() - t0:.0f} s")
    return video, audio


def write_mp4(path: str, frames, audio, sample_rate: int) -> None:
    """Writes the frames, in uint8 RGB, and the stereo audio of a clip to the mp4 `path`, in H.264 and MP3.

    diffusers' encode_video writes the audio in AAC, which the VS Code preview does not play.
    """
    import av

    with av.open(path, mode="w") as container:
        video_stream = container.add_stream("libx264", rate=int(FPS))
        video_stream.width, video_stream.height, video_stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        audio_stream = container.add_stream("libmp3lame", rate=sample_rate)
        audio_stream.codec_context.layout = "stereo"
        for array in frames:
            container.mux(video_stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(video_stream.encode())
        samples = av.AudioFrame.from_ndarray(audio, format="fltp", layout="stereo")
        samples.sample_rate = sample_rate
        samples.pts = 0
        container.mux(audio_stream.encode(samples))
        container.mux(audio_stream.encode())


def save(frames, audio, sample_rate: int, name: str, npz: bool = False) -> None:
    """Writes the clip of `frames`, in uint8 RGB, and `audio` to outputs/: NAME.mp4, NAME.png (4 of its frames) and,
    with `npz`, NAME.npz (its frames and audio)."""
    import numpy as np
    from PIL import Image

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    write_mp4(f"{path}.mp4", frames, audio, sample_rate)
    picks = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    # With the default compression, the 4 frames take 3 s to write.
    Image.fromarray(np.concatenate(list(frames[picks]), axis=1)).save(f"{path}.png", compress_level=1)
    if npz:
        np.savez(f"{path}.npz", video=frames, audio=audio)


def read_film(path: str, num_frames: int) -> list[dict]:
    """The shots of the film of the JSON file `path`, a list of shots of `num_frames` frames. Each has a "prompt", can
    start from the last frames of an earlier shot, {"from": {"shot": NUMBER, "frames": COUNT}}, which it then keeps as
    its first frames, and can end on the first frames of a later shot, {"to": {"shot": NUMBER, "frames": COUNT}}. As the
    VAE encodes the frames after the first by 8, COUNT is one more than a multiple of 8. A shot can also have its own
    "seed", in place of the seed of the film."""
    import json

    with open(path) as f:
        shots = json.load(f)
    if not shots:
        raise SystemExit(f"{path}: the film has no shot")
    for number, shot in enumerate(shots, 1):
        start, end = shot.get("from"), shot.get("to")
        if start is not None and not (
            1 <= start["shot"] < number and 1 <= start["frames"] <= num_frames and start["frames"] % 8 == 1
        ):
            raise SystemExit(f"{path}: shot {number} can only start from the last 1, 9, 17... frames of an earlier shot")
        # The keyframes start after the first latent of the clip.
        if end is not None and not (
            number < end["shot"] <= len(shots) and 1 <= end["frames"] <= num_frames - 8 and end["frames"] % 8 == 1
        ):
            raise SystemExit(f"{path}: shot {number} can only end on the first 1, 9, 17... frames of a later shot")
        if not isinstance(shot.get("seed", 0), int):
            raise SystemExit(f"{path}: the seed of shot {number} is not an integer")
        caption = shot.get("caption")
        if caption is not None and not (
            isinstance(caption.get("text"), str)
            and caption["text"]
            and 0 <= caption.get("start", -1) < caption.get("end", -1) <= num_frames / FPS
        ):
            raise SystemExit(f"{path}: the caption of shot {number} needs a text, a start and an end within the shot")
        trim = shot.get("trim")
        if trim is not None and not (0 <= trim.get("start", -1) < trim.get("end", -1) <= num_frames / FPS):
            raise SystemExit(f"{path}: the trim of shot {number} needs a start and an end within the shot")
    return shots


def generation_order(shots: list[dict]) -> list[int]:
    """The indices of `shots` in an order that makes each shot after those it starts from or ends on."""
    order = []
    while len(order) < len(shots):
        ready = [
            i
            for i, shot in enumerate(shots)
            if i not in order and all(shot[key]["shot"] - 1 in order for key in ("from", "to") if key in shot)
        ]
        if not ready:
            raise SystemExit("the film has shots that start from or end on each other")
        order.append(ready[0])
    return order


def draw_caption(frames, text: str) -> None:
    """Draws `text` on `frames`, in uint8 RGB, in place: centered near the bottom, in white letters outlined in black,
    a fifth of the frame high, in the font that comes with Pillow."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    height, width = frames.shape[1:3]
    font, stroke = ImageFont.load_default(size=height // 5), height // 64
    left, _, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font, stroke_width=stroke)
    position = ((width - left - right) // 2, height - height // 12 - bottom)
    for frame in frames:
        image = Image.fromarray(frame)
        ImageDraw.Draw(image).text(position, text, "white", font, stroke_width=stroke, stroke_fill="black")
        frame[:] = np.asarray(image)


def level_audio(audio, sample_rate: int):
    """`audio`, stereo, with its quiet parts raised toward -52 dBFS over half a second, by at most 18 dB, and with 3 dB
    of headroom left under the loud sounds within 0.4 s of them. The model generates the sound of each shot of a film
    on its own, its background at a level of its own, from almost silent to clearly audible."""
    import numpy as np

    hop = sample_rate // 100
    count = -(-audio.shape[1] // hop)
    hops = np.pad(audio, ((0, 0), (0, count * hop - audio.shape[1]))).reshape(2, count, hop)
    power = (hops.astype(np.float64) ** 2).mean(axis=(0, 2))
    level = 10 * np.log10(np.convolve(np.pad(power, 25, mode="edge"), np.ones(51) / 51, mode="valid") + 1e-12)
    peak = np.lib.stride_tricks.sliding_window_view(np.pad(np.abs(hops).max(axis=(0, 2)), 40, mode="edge"), 81)
    headroom = np.maximum(-3 - 20 * np.log10(peak.max(axis=1) + 1e-12), 0)
    gain = np.minimum(np.clip(-52 - level, 0, 18), headroom)
    gain = np.convolve(np.pad(gain, 10, mode="edge"), np.ones(21) / 21, mode="valid")
    scale = np.interp(np.arange(audio.shape[1]), (np.arange(count) + 0.5) * hop, 10 ** (gain / 20))
    return (audio * scale).astype(audio.dtype)


def edit_film(clips: list, shots: list[dict], sample_rate: int):
    """The frames and audio of the film of `shots`, of which `clips` holds the frames and audio: its shots one after the
    other, each without the frames it starts from, which repeat those of an earlier shot, and without its frames from
    the keyframes it ends on, which a later shot repeats. A shot can be trimmed to a time range, and have a caption,
    drawn on its frames from its start to its end, both in seconds into the shot. Its sound has the background of its
    shots evened out (see `level_audio`)."""
    import numpy as np

    frames, audio = [], []
    for (shot_frames, shot_audio), shot in zip(clips, shots):
        skip, stop = shot.get("from", {}).get("frames", 0), len(shot_frames)
        if "to" in shot:
            stop = (keyframe_index(len(shot_frames), shot["to"]["frames"]) - 1) * 8 + 1
        if "trim" in shot:
            skip, stop = max(skip, round(shot["trim"]["start"] * FPS)), min(stop, round(shot["trim"]["end"] * FPS))
        kept = shot_frames[skip:stop]
        if "caption" in shot:
            first, last = (round(shot["caption"][key] * FPS) - skip for key in ("start", "end"))
            kept = kept.copy()
            draw_caption(kept[max(first, 0) : max(last, 0)], shot["caption"]["text"])
        frames.append(kept)
        # The audio of a clip ends a little before its frames (32 ms with the vocoder of LTX-2.5): padded to their
        # length, each shot keeps the film in sync.
        offset, length = round(skip / FPS * sample_rate), round((stop - skip) / FPS * sample_rate)
        shot_audio = shot_audio[:, offset : offset + length]
        audio.append(np.pad(shot_audio, ((0, 0), (0, length - shot_audio.shape[1]))))
    return np.concatenate(frames), level_audio(np.concatenate(audio, axis=1), sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generates a clip with LTX-2.5, on Neuron or on CPU.")
    parser.add_argument("--cpu", action="store_true", help="run the whole pipeline on CPU")
    content = parser.add_mutually_exclusive_group()
    content.add_argument("--prompt", nargs="+", default=[PROMPT], help="one clip per prompt, made one after the other")
    content.add_argument("--film", help="a JSON file of the shots of a film, which can start from the last frames of "
                         "an earlier shot (see read_film): the shots, made one after the other, then the film")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--npz", action="store_true", help="also write the frames and audio of each clip to "
                        "NAME.npz")
    parser.add_argument("--name", help="the name of the outputs (default: video-gen-demo-WIDTHxHEIGHT, after the size "
                        "of the clip, or FILM-WIDTHxHEIGHT with --film, after its file, with -cpu on CPU), followed by "
                        "-1, -2... for each of several clips, or by -shot-1, -shot-2... for the shots of a film and "
                        "-full for the film")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES, help="one more than a multiple of 8")
    parser.add_argument("--two-stage", action="store_true", help="upscale the clip x2 with the two-stage recipe of "
                        "the model card")
    parser.add_argument("--tp", type=int, default=TP, help="the number of NeuronCores the weights are split over")
    parser.add_argument("--sp", type=int, help="the number of groups of --tp NeuronCores that split the video tokens "
                        "(default: 4 with --two-stage, else 1)")
    args = parser.parse_args()
    shots = read_film(args.film, args.frames) if args.film else [{"prompt": prompt} for prompt in args.prompt]
    order = generation_order(shots)
    # Split over 4 cores only, the block graph of the second stage fails to load.
    args.sp = args.sp or (4 if args.two_stage else 1)
    if args.cpu:
        # Keeps torch from loading torch-neuronx: this run takes no NeuronCore.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    else:
        launch.set_neuron_environment(args.tp * args.sp)
        if "LOCAL_RANK" not in os.environ:
            raise SystemExit(launch.torchrun(__file__, args.tp * args.sp, sys.argv[1:]))

    calls = []
    t0 = time.time()
    transformer = vae = group = sp_group = None
    if not args.cpu:
        transformer, vae, group, sp_group = load_models(args.sp)
        if RANK != 0:
            serve(transformer, vae, group, sp_group)
            return
    pipe = load_pipeline(calls, transformer, vae, group, sp_group)
    upsample_pipe = load_upsampler(pipe) if args.two_stage else None
    conditioned = any("from" in shot or "to" in shot for shot in shots)
    condition_pipe = load_condition_pipeline(pipe) if conditioned else None
    log(f"load: {time.time() - t0:.0f} s")
    import torch

    sample_rate = pipe.vocoder.config.output_sampling_rate
    # The frames, in uint8 RGB, and the audio of each clip, once made.
    clips = [None] * len(shots)
    for i in order:
        shot = shots[i]
        calls.clear()
        t0 = time.time()
        start = end = None
        if "from" in shot:
            source, count = shot["from"]["shot"], shot["from"]["frames"]
            log(f"shot {i + 1}: starts from the last {count} frames of shot {source}")
            start = clips[source - 1][0][-count:]
        if "to" in shot:
            target, count = shot["to"]["shot"], shot["to"]["frames"]
            log(f"shot {i + 1}: ends on the first {count} frames of shot {target}")
            end = clips[target - 1][0][:count]
        video, audio = generate(
            pipe if start is None and end is None else condition_pipe,
            shot["prompt"],
            shot.get("seed", args.seed),
            args.width,
            args.height,
            args.frames,
            upsample_pipe,
            start,
            end,
        )
        if group is not None and i == order[-1]:
            import torch.distributed as dist

            # The other ranks return from `serve`.
            dist.broadcast_object_list([None], src=0, group=group)
        log(f"generation: {time.time() - t0:.0f} s, transformer calls: {', '.join(f'{c:.1f}' for c in calls)} s")
        # On all the threads of torch, in place of numpy, which takes 5 s on one: the same rounding, the same bytes.
        frames = video[0].float().mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
        clips[i] = (frames, audio[0].float().cpu().numpy())
        height, width = frames.shape[1:3]
        stem = os.path.splitext(os.path.basename(args.film))[0] if args.film else "video-gen-demo"
        name = args.name or f"{stem}-{width}x{height}" + ("-cpu" if args.cpu else "")
        if args.film:
            clip_name = f"{name}-shot-{i + 1}"
        else:
            clip_name = f"{name}-{i + 1}" if len(shots) > 1 else name
        save(*clips[i], sample_rate, clip_name, npz=args.npz)
        log(f"wrote {os.path.join(OUTPUT_DIR, clip_name)}.mp4, {time.time() - t0:.0f} s after the start of the clip")
    if args.film:
        t0 = time.time()
        save(*edit_film(clips, shots, sample_rate), sample_rate, f"{name}-full")
        log(f"wrote {os.path.join(OUTPUT_DIR, name)}-full.mp4, the film, in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
