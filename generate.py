"""Generates a clip with its audio with LTX-2.5, its transformer split over 2 NeuronCores, or all on CPU.

The distilled pipeline of models/LTX-2.5-Diffusers generates in one stage, with the settings of its model card. On
Neuron, `parallel.load_transformer` splits the transformer over the first 2 NeuronCores of NEURON_RT_VISIBLE_CORES,
one torch.distributed rank per core, and torch.compile(backend="neuron") compiles it. Every rank runs the other models
on CPU. With --cpu, the whole pipeline runs on CPU.

The clip goes to outputs/NAME.mp4, 4 of its frames to outputs/NAME.png, and its frames and audio to outputs/NAME.npz.

Run with:
    python generate.py [--cpu] [--prompt PROMPT] [--seed SEED] [--name NAME]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import launch

MODEL_DIR = os.path.join(launch.REPO_DIR, "models", "LTX-2.5-Diffusers")
OUTPUT_DIR = os.path.join(launch.REPO_DIR, "outputs")
TP = 2
RANK = int(os.environ.get("RANK", "0"))
# 2 seconds at 768x512: LTX-2.5 takes one frame more than a multiple of 8.
WIDTH, HEIGHT, NUM_FRAMES, FPS = 768, 512, 49, 24.0
PROMPT = (
    "A red fox trots through fresh snow in a sunlit pine forest, its breath steaming in the cold air. The camera "
    "tracks alongside it at ground level. Snow crunches under its paws and birds sing in the distance."
)


def log(msg: str) -> None:
    if RANK == 0:
        print(f"[generate] {msg}", flush=True)


def timed(transformer, calls: list, device: str | None) -> None:
    """Makes `transformer` append the duration of each of its calls to `calls`. On `device` if given, compiled for
    it, it takes and returns tensors on the host."""
    import torch
    from torch.utils._pytree import tree_map

    forward = transformer.forward
    if device is not None:
        forward = torch.compile(forward, backend="neuron", fullgraph=False, dynamic=False)

    def call(*args, **kwargs):
        t0 = time.time()
        if device is not None:
            args, kwargs = tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, (args, kwargs))
        output = tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, forward(*args, **kwargs))
        calls.append(time.time() - t0)
        return output

    transformer.forward = call


def load_pipeline(cpu: bool, calls: list):
    """The distilled LTX-2.5 pipeline in bf16, its tensors on the host, and its transformer on Neuron unless `cpu`."""
    import torch
    from diffusers import LTX2Pipeline

    class HostPipeline(LTX2Pipeline):
        # diffusers runs a pipeline on the device of its accelerated models. All
        # models but the transformer run on the host, and the transformer moves
        # its own inputs.
        @property
        def _execution_device(self):
            return torch.device("cpu")

    components = {}
    if not cpu:
        import torch.distributed as dist
        from torch.distributed.device_mesh import DeviceMesh

        import parallel

        dist.init_process_group(backend="neuron")
        mesh = DeviceMesh("neuron", list(range(dist.get_world_size())))
        components["transformer"] = parallel.load_transformer(os.path.join(MODEL_DIR, "transformer"), mesh)
    # The processor and the prompt enhancer serve prompt enhancement only.
    pipe = HostPipeline.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, processor=None, prompt_enhancer=None, **components
    )
    timed(pipe.transformer, calls, None if cpu else f"neuron:{torch.neuron.current_device()}")
    return pipe


def generate(pipe, prompt: str, seed: int):
    """The video, as frames in [0, 1], and the audio of the clip of `prompt`: those of rank 0, latents on the others."""
    import torch
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES

    return pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        frame_rate=FPS,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        # Drawn on the host, the noise is the same on every rank and on CPU.
        generator=torch.Generator("cpu").manual_seed(seed),
        # The ranks run the transformer together, then rank 0 decodes alone.
        output_type="np" if RANK == 0 else "latent",
        return_dict=False,
    )


def save(video, audio, sample_rate: int, name: str) -> None:
    """Writes the clip to outputs/: NAME.mp4, NAME.png (4 of its frames) and NAME.npz (its frames and audio)."""
    import numpy as np
    from diffusers.utils import encode_video
    from PIL import Image

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    encode_video(video, fps=int(FPS), output_path=f"{path}.mp4", audio=audio, audio_sample_rate=sample_rate)
    frames = (np.clip(video, 0, 1) * 255).round().astype(np.uint8)
    picks = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    Image.fromarray(np.concatenate(list(frames[picks]), axis=1)).save(f"{path}.png")
    np.savez(f"{path}.npz", video=frames, audio=audio.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generates a clip with LTX-2.5, on Neuron or on CPU.")
    parser.add_argument("--cpu", action="store_true", help="run the whole pipeline on CPU")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="the name of the outputs (default: neuron, or cpu with --cpu)")
    args = parser.parse_args()
    if args.cpu:
        # Keeps torch from loading torch-neuronx: this run takes no NeuronCore.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    else:
        launch.set_neuron_environment(TP)
        if "LOCAL_RANK" not in os.environ:
            raise SystemExit(launch.torchrun(__file__, TP, sys.argv[1:]))

    calls = []
    t0 = time.time()
    pipe = load_pipeline(args.cpu, calls)
    log(f"load: {time.time() - t0:.0f} s")
    t0 = time.time()
    video, audio = generate(pipe, args.prompt, args.seed)
    log(f"generation: {time.time() - t0:.0f} s, transformer: first call {calls[0]:.1f} s, "
        f"then {sum(calls[1:]) / max(len(calls) - 1, 1):.2f} s per call over {len(calls) - 1}")
    if RANK == 0:
        name = args.name or ("cpu" if args.cpu else "neuron")
        save(video[0], audio[0].float().cpu(), pipe.vocoder.config.output_sampling_rate, name)
        log(f"wrote outputs/{name}.mp4")


if __name__ == "__main__":
    main()
