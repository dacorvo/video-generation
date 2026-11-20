"""Checks on CPU that the LTX-2.5 video VAE split by frames by `parallel.split_by_frames` decodes what it decodes whole.

A video VAE of the LTX-2.5 architecture, narrower and with random weights,
decodes random latents of 11 frames whole, then split over 4 ranks, as 3, 3,
3 and 2 frames, in fp32. Put together, the frames of the ranks must be those
of the whole decode, to fp32 rounding.

Run with:
    ./run.sh check_frame_split.py
"""
from __future__ import annotations

import os
import socket

# Keeps torch from loading torch-neuronx, in this process and the ranks it
# spawns: this check must not take NeuronCores.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WORLD_SIZE = 4
# The config of the LTX-2.5 video VAE, at a fraction of its sizes: its
# settings that differ from the defaults of diffusers, with fewer and narrower
# layers.
CONFIG = dict(
    latent_channels=8,
    block_out_channels=(8, 16, 32, 32),
    decoder_block_out_channels=(16, 32, 32, 64),
    layers_per_block=(1, 1, 1, 1, 1),
    decoder_layers_per_block=(1, 1, 1, 1, 1),
    decoder_spatio_temporal_scaling=(True, True, True, True),
    decoder_inject_noise=(False, False, False, False, False),
    upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    upsample_residual=(False, False, False, False),
    upsample_factor=(2, 2, 1, 2),
    decoder_causal=False,
    decoder_spatial_padding_mode="zeros",
    spatial_compression_ratio=32,
    temporal_compression_ratio=8,
)
TOLERANCE = 1e-5


def run(rank: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    from diffusers import AutoencoderKLLTX2Video

    import parallel

    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(**CONFIG)
    latents = torch.randn(1, CONFIG["latent_channels"], 11, 4, 6, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        whole = vae.decode(latents, None, return_dict=False)[0]
        parallel.split_by_frames(vae, dist.group.WORLD)
        frames = vae.decode(latents, None, return_dict=False)[0]
    parts = [None] * WORLD_SIZE if rank == 0 else None
    dist.gather_object(frames, parts, dst=0)
    if rank == 0:
        split = torch.cat(parts, dim=2)
        error = ((split - whole).norm() / whole.norm()).item() if split.shape == whole.shape else float("inf")
        print(f"frames per rank {[p.shape[2] for p in parts]}, {whole.shape[2]} frames whole: relative error {error:.2e}")
        print("PASS" if error < TOLERANCE else f"FAIL: above {TOLERANCE:.0e}")
    dist.destroy_process_group()


def main() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(run, args=(port,), nprocs=WORLD_SIZE)


if __name__ == "__main__":
    main()
