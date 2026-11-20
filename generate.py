"""Generates a clip with its audio with LTX-2.5, its transformer split over NeuronCores, or all on CPU.

The distilled pipeline of models/LTX-2.5-Diffusers generates in one stage, with the settings of its model card. On
Neuron, `parallel` splits the transformer, and the decoder of the video VAE by frames, over the first --tp NeuronCores
of NEURON_RT_VISIBLE_CORES, one torch.distributed rank per core, and torch.compile(backend="neuron") compiles the
transformer's blocks and the decoder. Only rank 0 runs the pipeline, its other models on CPU. It sends the arguments of
each transformer call and of the decode to the other ranks, which run their slices of the transformer on them and
decode their frames, which rank 0 gathers. With --cpu, the whole pipeline runs on CPU.

The clip goes to outputs/NAME.mp4, 4 of its frames to outputs/NAME.png, and its frames and audio to outputs/NAME.npz.
outputs/ is in the repository, and models/ in the working directory of the demo, `launch.WORK_DIR`.

Run with:
    ./run.sh generate.py [--cpu] [--prompt PROMPT] [--seed SEED] [--name NAME] [--width WIDTH] [--height HEIGHT]
        [--frames FRAMES] [--tp TP]
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
TP = 2
RANK = int(os.environ.get("RANK", "0"))
# By default, 2 seconds at 768x512. LTX-2.5 takes one frame more than a multiple of 8.
WIDTH, HEIGHT, NUM_FRAMES, FPS = 768, 512, 49, 24.0
PROMPT = (
    "A red fox trots through fresh snow in a sunlit pine forest, its breath steaming in the cold air. The camera "
    "tracks alongside it at ground level. Snow crunches under its paws and birds sing in the distance."
)


def log(msg: str) -> None:
    if RANK == 0:
        print(f"[generate] {msg}", flush=True)


def on_neuron(transformer):
    """The forward of `transformer`, its blocks compiled for the NeuronCore of this rank: it takes and returns tensors
    on the host."""
    import torch
    from torch.utils._pytree import tree_map

    device = f"neuron:{torch.neuron.current_device()}"
    # The blocks share one compiled graph, so a new process traces and lowers a single block. The rest of the forward
    # runs op by op.
    transformer.compile_repeated_blocks(backend="neuron", fullgraph=True, dynamic=False)
    forward = transformer.forward

    def call(*args, **kwargs):
        args, kwargs = tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, (args, kwargs))
        return tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, forward(*args, **kwargs))

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


def timed(transformer, calls: list, group=None) -> None:
    """Makes `transformer` append the duration of each of its calls to `calls`.

    With `group`, the gloo group of the ranks that split `transformer` over their NeuronCores, it runs on the core of
    this rank, after it sends the arguments of the call to the other ranks, which run their slices on them (`serve`).
    """
    import torch.distributed as dist

    forward = transformer.forward if group is None else on_neuron(transformer)

    def call(*args, **kwargs):
        t0 = time.time()
        if group is not None:
            dist.broadcast_object_list([("transformer", args, kwargs)], src=0, group=group)
        output = forward(*args, **kwargs)
        calls.append(time.time() - t0)
        return output

    transformer.forward = call


def serve(transformer, vae, group) -> None:
    """Runs on the NeuronCore of this rank, on the arguments that rank 0 sends over `group`, the slice of `transformer`
    of this rank for each transformer call (see `timed`), and the decoder of `vae` on the frames of this rank for each
    decode, which it then sends to rank 0 (see `split_decode`), until it sends None."""
    import torch
    import torch.distributed as dist

    forward = on_neuron(transformer)
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


def load_models():
    """The slice of the LTX-2.5 transformer of this rank, and the video VAE, its decoder on the NeuronCore of this rank,
    both split over the NeuronCores of all ranks, and a gloo group of all ranks, which sends the arguments of their
    calls."""
    import torch
    import torch.distributed as dist
    from diffusers import AutoencoderKLLTX2Video
    from torch.distributed.device_mesh import DeviceMesh

    import parallel

    dist.init_process_group(backend="neuron")
    mesh = DeviceMesh("neuron", list(range(dist.get_world_size())))
    transformer = parallel.load_transformer(os.path.join(MODEL_DIR, "transformer"), mesh)
    vae = AutoencoderKLLTX2Video.from_pretrained(os.path.join(MODEL_DIR, "vae"), dtype=torch.bfloat16)
    # The pipeline normalizes the latents on the host with the other tensors of the VAE.
    vae.decoder.to(f"neuron:{torch.neuron.current_device()}")
    parallel.split_by_frames(vae, mesh.get_group())
    return transformer, vae, dist.new_group(backend="gloo")


def load_pipeline(calls: list, transformer=None, vae=None, group=None):
    """The distilled LTX-2.5 pipeline in bf16, its tensors on the host. Its transformer and video VAE run on CPU, or
    are `transformer` and `vae` on Neuron, with `group`, as `load_models` returns them."""
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
    timed(pipe.transformer, calls, group)
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
    parser.add_argument("--tp", type=int, default=TP, help="the number of NeuronCores the models are split over")
    args = parser.parse_args()
    if args.cpu:
        # Keeps torch from loading torch-neuronx: this run takes no NeuronCore.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    else:
        launch.set_neuron_environment(args.tp)
        if "LOCAL_RANK" not in os.environ:
            raise SystemExit(launch.torchrun(__file__, args.tp, sys.argv[1:]))

    calls = []
    t0 = time.time()
    transformer = vae = group = None
    if not args.cpu:
        transformer, vae, group = load_models()
        if RANK != 0:
            serve(transformer, vae, group)
            return
    pipe = load_pipeline(calls, transformer, vae, group)
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
