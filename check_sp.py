"""Checks on CPU that the LTX-2.5 transformer split over 2 x 2 ranks by `parallel.py` computes what it computes whole.

The transformer of check_tp.py, of the LTX-2.5 architecture, smaller and with
random weights, runs whole, then over 4 ranks, on the same inputs, in fp32: 2
groups of 2 ranks split its video tokens, and the 2 ranks of each group split
its weights. The outputs must match to fp32 rounding, with one timestep for the
clip, and with one per video token, as for a clip that starts from frames.

Run with:
    ./run.sh check_sp.py
"""
from __future__ import annotations

import os
import socket
import tempfile

# Keeps torch from loading torch-neuronx, in this process and the ranks it
# spawns: this check must not take NeuronCores.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from check_tp import CONFIG, TOLERANCE, inputs, relative_error  # noqa: E402

TP, SP = 2, 2


def per_token(kwargs: dict) -> dict:
    """`kwargs`, the inputs of `inputs`, with a timestep per video token, random so that a wrong slice cannot match, and
    the timestep of the clip for the audio, as the pipeline passes them for a clip that starts from frames."""
    batch_size, tokens = kwargs["hidden_states"].shape[:2]
    timestep = torch.rand(batch_size, tokens, generator=torch.Generator().manual_seed(1)) * 1000
    return {**kwargs, "timestep": timestep, "audio_timestep": kwargs["timestep"]}


def run(rank: int, path: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=TP * SP)
    import diffusers.hooks.tensor_parallel
    from diffusers import LTX2VideoTransformer3DModel
    from torch.distributed.device_mesh import DeviceMesh

    import parallel

    # diffusers splits models on accelerators only, DTensor runs on CPU too.
    # Without an accelerator, the current one is the CPU, of which diffusers
    # needs a device count.
    diffusers.hooks.tensor_parallel._SUPPORTED_TP_DEVICES += ("cpu",)
    torch.accelerator.device_count = lambda: 1
    mesh = DeviceMesh("cpu", torch.arange(TP * SP).reshape(SP, TP), mesh_dim_names=("sp", "tp"))
    sp_group = mesh["sp"].get_group()
    whole = LTX2VideoTransformer3DModel.from_pretrained(path)
    split = parallel.load_transformer(path, mesh["tp"], sp_group, dtype=torch.float32)
    worst = 0.0
    for case, kwargs in (("timestep per clip", inputs()), ("timestep per token", per_token(inputs()))):
        # A transformer split by its video tokens takes their positions as arguments.
        kwargs["video_coords"] = whole.rope.prepare_video_coords(
            kwargs["hidden_states"].shape[0], kwargs["num_frames"], kwargs["height"], kwargs["width"], torch.device("cpu")
        )
        with torch.no_grad():
            expected = whole(**kwargs)
            video, audio = split(**parallel.split_video_tokens(kwargs, sp_group))
            actual = (parallel.gather_video_tokens(video, sp_group), audio)
        if rank == 0:
            errors = {name: relative_error(a, e) for name, a, e in zip(("video", "audio"), actual, expected)}
            print(f"{case}: " + ", ".join(f"{name}: relative error {error:.2e}" for name, error in errors.items()))
            worst = max(worst, *errors.values())
    if rank == 0:
        print("PASS" if worst < TOLERANCE else f"FAIL: above {TOLERANCE:.0e}")
    dist.destroy_process_group()


def main() -> None:
    from diffusers import LTX2VideoTransformer3DModel

    torch.manual_seed(0)
    model = LTX2VideoTransformer3DModel(**CONFIG)
    # Off their initial values, so that a norm weight of ones or a bias of
    # zeros cannot hide a wrong slice.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1 * torch.randn_like(param))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as path:
        model.save_pretrained(path)
        mp.spawn(run, args=(path, port), nprocs=TP * SP)


if __name__ == "__main__":
    main()
