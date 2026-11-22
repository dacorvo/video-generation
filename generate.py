"""Generates a clip with its audio with LTX-2.5, its transformer split over NeuronCores, or all on CPU.

The distilled pipeline of models/LTX-2.5-Diffusers generates in one stage, with the settings of its model card. On
Neuron, `parallel` splits the transformer, and the decoder of the video VAE by frames, over the first --tp NeuronCores
of NEURON_RT_VISIBLE_CORES, one torch.distributed rank per core, and torch.compile(backend="neuron") compiles the
transformer's blocks and the decoder. With --sp, --sp groups of --tp cores split the video tokens of the transformer,
and the decoder runs on all of their cores. Only rank 0 runs the pipeline, its other models on CPU. It sends the
arguments of each transformer call and of the decode to the other ranks, which run their slices of the transformer on
them and decode their frames, which rank 0 gathers. With --cpu, the whole pipeline runs on CPU.

The clip goes to outputs/NAME.mp4, 4 of its frames to outputs/NAME.png, and its frames and audio to outputs/NAME.npz.
outputs/ is in the repository, and models/ in the working directory of the demo, `launch.WORK_DIR`.

Run with:
    ./run.sh generate.py [--cpu] [--prompt PROMPT] [--seed SEED] [--name NAME] [--width WIDTH] [--height HEIGHT]
        [--frames FRAMES] [--tp TP] [--sp SP]
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
    and returns the video output of all of them."""
    import torch
    from torch.utils._pytree import tree_map

    import parallel

    device = f"neuron:{torch.neuron.current_device()}"
    # The blocks share one compiled graph, so a new process traces and lowers a single block. The rest of the forward
    # runs op by op.
    transformer.compile_repeated_blocks(backend="neuron", fullgraph=True, dynamic=False)
    forward = transformer.forward

    def call(*args, **kwargs):
        if sp_group is not None:
            kwargs = parallel.split_video_tokens(kwargs, sp_group)
        args, kwargs = tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, (args, kwargs))
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
    vae.decoder.compile(backend="neuron", fullgraph=True, dynamic=False)
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


def load_pipeline(calls: list, transformer=None, vae=None, group=None, sp_group=None):
    """The distilled LTX-2.5 pipeline in bf16, its tensors on the host. Its transformer and video VAE run on CPU, or
    are `transformer` and `vae` on Neuron, with `group` and `sp_group`, as `load_models` returns them."""
    import torch
    from diffusers import LTX2Pipeline

    class HostPipeline(LTX2Pipeline):
        # diffusers runs a pipeline on the device of its accelerated models. All
        # models but the transformer run on the host, and the transformer moves
        # its own inputs.
        @property
        def _execution_device(self):
            return torch.device("cpu")

    components = {} if transformer is None else {"transformer": transformer, "vae": vae}
    # The processor and the prompt enhancer serve prompt enhancement only.
    pipe = HostPipeline.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, processor=None, prompt_enhancer=None, **components
    )
    timed(pipe.transformer, calls, group, sp_group)
    if group is not None:
        split_decode(pipe.vae, group)
    return pipe


def generate(pipe, prompt: str, seed: int, width: int, height: int, num_frames: int):
    """The video, as frames in [0, 1], and the audio of the clip of `prompt`."""
    import torch
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES

    return pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=FPS,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        # Drawn on the host, the noise is the same on Neuron and on CPU.
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
        return_dict=False,
    )


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


def save(video, audio, sample_rate: int, name: str) -> None:
    """Writes the clip to outputs/: NAME.mp4, NAME.png (4 of its frames) and NAME.npz (its frames and audio)."""
    import numpy as np
    from PIL import Image

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    frames = (np.clip(video, 0, 1) * 255).round().astype(np.uint8)
    audio = audio.numpy()
    write_mp4(f"{path}.mp4", frames, audio, sample_rate)
    picks = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    Image.fromarray(np.concatenate(list(frames[picks]), axis=1)).save(f"{path}.png")
    np.savez(f"{path}.npz", video=frames, audio=audio)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generates a clip with LTX-2.5, on Neuron or on CPU.")
    parser.add_argument("--cpu", action="store_true", help="run the whole pipeline on CPU")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="the name of the outputs (default: neuron, or cpu with --cpu)")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES, help="one more than a multiple of 8")
    parser.add_argument("--tp", type=int, default=TP, help="the number of NeuronCores the weights are split over")
    parser.add_argument("--sp", type=int, default=1, help="the number of groups of --tp NeuronCores that split the "
                        "video tokens")
    args = parser.parse_args()
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
    log(f"load: {time.time() - t0:.0f} s")
    t0 = time.time()
    video, audio = generate(pipe, args.prompt, args.seed, args.width, args.height, args.frames)
    if group is not None:
        import torch.distributed as dist

        # The other ranks return from `serve`.
        dist.broadcast_object_list([None], src=0, group=group)
    log(f"generation: {time.time() - t0:.0f} s, transformer: first call {calls[0]:.1f} s, "
        f"then {sum(calls[1:]) / max(len(calls) - 1, 1):.2f} s per call over {len(calls) - 1}")
    name = args.name or ("cpu" if args.cpu else "neuron")
    save(video[0], audio[0].float().cpu(), pipe.vocoder.config.output_sampling_rate, name)
    log(f"wrote {os.path.join(OUTPUT_DIR, name)}.mp4")


if __name__ == "__main__":
    main()
