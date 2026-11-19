"""Checks that the LTX-2.5 transformer split over 2 NeuronCores computes what it computes on CPU.

`parallel.load_transformer` splits the transformer of models/LTX-2.5-Diffusers
over the first 2 NeuronCores of NEURON_RT_VISIBLE_CORES, one torch.distributed
rank per core, and torch.compile(backend="neuron") compiles it. It runs on the
inputs of one denoising step of a 1-second clip at 768x512, and so does the
whole transformer on CPU, in bf16. As in the model support sweep, the error
allowed is twice the gap between the CPU outputs in bf16 and in fp32, and at
least 0.02. models/ is in the working directory of the demo, `launch.WORK_DIR`.

Run with:
    ./run.sh check_neuron.py
"""
from __future__ import annotations

import os
import time
import traceback

import launch

TRANSFORMER_DIR = os.path.join(launch.WORK_DIR, "models", "LTX-2.5-Diffusers", "transformer")
TP = 2
RANK = int(os.environ.get("RANK", "0"))
launch.set_neuron_environment(TP)

# The latents of 25 frames at 768x512 (32x spatial, 8x temporal compression
# after the first frame) and of their 1.04 s of audio (25 latents per second).
FRAMES, HEIGHT, WIDTH = 4, 16, 24
AUDIO_FRAMES = 26
TEXT_LENGTH = 128
# A timestep of the distilled schedule, in [0, 1000].
TIMESTEP = 900.0
TOLERANCE = 0.02


def banner(msg: str) -> None:
    if RANK == 0:
        print(f"[check] {msg}", flush=True)


def inputs(config, dtype) -> dict:
    """The transformer's inputs at one step, rounded to bf16 then cast to `dtype`, on the host: the same at every call."""
    import torch

    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).bfloat16().to(dtype)

    inner_dim = config.num_attention_heads * config.attention_head_dim
    audio_inner_dim = config.audio_num_attention_heads * config.audio_attention_head_dim
    timestep = torch.tensor([TIMESTEP])
    mask = torch.ones(1, TEXT_LENGTH)
    return dict(
        hidden_states=randn(1, FRAMES * HEIGHT * WIDTH, config.in_channels),
        audio_hidden_states=randn(1, AUDIO_FRAMES, config.audio_in_channels),
        encoder_hidden_states=randn(1, TEXT_LENGTH, inner_dim),
        audio_encoder_hidden_states=randn(1, TEXT_LENGTH, audio_inner_dim),
        timestep=timestep,
        sigma=timestep,
        encoder_attention_mask=mask,
        audio_encoder_attention_mask=mask,
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        audio_num_frames=AUDIO_FRAMES,
        use_cross_timestep=True,
        return_dict=False,
    )


def moved(args: dict, device: str) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in args.items()}


def relative_error(output, reference) -> float:
    output, reference = output.float(), reference.float()
    return ((output - reference).norm() / reference.norm()).item()


def run() -> int:
    """Runs the split transformer on this rank's core, then compares on rank 0. Returns 0 on PASS, 1 on FAIL."""
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh

    import parallel

    dist.init_process_group(backend="neuron")
    mesh = DeviceMesh("neuron", list(range(dist.get_world_size())))
    device = f"neuron:{torch.neuron.current_device()}"
    try:
        t0 = time.time()
        transformer = parallel.load_transformer(TRANSFORMER_DIR, mesh)
        banner(f"load: {time.time() - t0:.0f} s")
        transformer.compile(backend="neuron", fullgraph=False, dynamic=False)
        args = moved(inputs(transformer.config, torch.bfloat16), device)
        with torch.no_grad():
            for call in ("first call, which compiles", "second call"):
                t0 = time.time()
                output = [o.cpu() for o in transformer(**args)]
                banner(f"{call}: {time.time() - t0:.1f} s")
    except Exception as exc:
        print(f"[check][rank{RANK}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    if RANK != 0:
        return 0

    from diffusers import LTX2VideoTransformer3DModel

    t0 = time.time()
    expected = {}
    for dtype in (torch.bfloat16, torch.float32):
        whole = LTX2VideoTransformer3DModel.from_pretrained(TRANSFORMER_DIR, dtype=dtype)
        with torch.no_grad():
            expected[dtype] = whole(**inputs(whole.config, dtype))
        del whole
    banner(f"cpu runs: {time.time() - t0:.0f} s")
    failed = False
    for i, name in enumerate(("video", "audio")):
        error = relative_error(output[i], expected[torch.bfloat16][i])
        gap = relative_error(expected[torch.bfloat16][i], expected[torch.float32][i])
        limit = max(TOLERANCE, 2 * gap)
        failed |= error > limit
        banner(f"{name}: error {error:.4f} from CPU, limit {limit:.4f} (bf16 vs fp32 on CPU: {gap:.4f}, "
               f"Neuron vs fp32 on CPU: {relative_error(output[i], expected[torch.float32][i]):.4f})")
    banner("FAIL" if failed else "PASS")
    return int(failed)


def main() -> None:
    if "LOCAL_RANK" in os.environ:
        # A rank that the torchrun below launched.
        raise SystemExit(run())
    raise SystemExit(launch.torchrun(__file__, TP, []))


if __name__ == "__main__":
    main()
