"""Checks on CPU that the LTX-2.5 transformer split by `parallel.load_transformer` computes what it computes whole.

A transformer of the LTX-2.5 architecture, smaller and with random weights, runs
whole, then split over 2 ranks, on the same inputs, in fp32. The outputs must
match to fp32 rounding: splitting changes only the order of some sums.

Run with:
    python check_tp.py
"""
from __future__ import annotations

import os
import socket
import tempfile

# Keeps torch from loading torch-neuronx, in this process and the ranks it
# spawns: diffusers places a split model on the current accelerator, and this
# check must not take NeuronCores.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WORLD_SIZE = 2
# The config of the LTX-2.5 transformer, at a fraction of its sizes: its
# settings that differ from the defaults of diffusers, such as gated attention,
# cross-attention modulation and split RoPE, with fewer and narrower layers.
CONFIG = dict(
    in_channels=16,
    out_channels=16,
    num_attention_heads=4,
    attention_head_dim=16,
    cross_attention_dim=64,
    gated_attn=True,
    cross_attn_mod=True,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_num_attention_heads=4,
    audio_attention_head_dim=8,
    audio_cross_attention_dim=32,
    audio_gated_attn=True,
    audio_cross_attn_mod=True,
    num_layers=2,
    rope_type="split",
    use_prompt_embeddings=False,
    ff_bias=False,
    perturbed_attn=True,
    use_keyframes_abs_pos_embedding=True,
)
TOLERANCE = 1e-5


def inputs(batch_size: int = 2) -> dict:
    """The transformer's inputs for a short clip, the second prompt padded."""
    generator = torch.Generator().manual_seed(0)
    frames, height, width, audio_frames, text_length = 2, 4, 4, 8, 6
    mask = torch.ones(batch_size, text_length)
    mask[1, -2:] = 0
    timestep = torch.full((batch_size,), 700.0)
    inner_dim = CONFIG["num_attention_heads"] * CONFIG["attention_head_dim"]
    audio_inner_dim = CONFIG["audio_num_attention_heads"] * CONFIG["audio_attention_head_dim"]
    return dict(
        hidden_states=torch.randn(batch_size, frames * height * width, CONFIG["in_channels"], generator=generator),
        audio_hidden_states=torch.randn(batch_size, audio_frames, CONFIG["audio_in_channels"], generator=generator),
        encoder_hidden_states=torch.randn(batch_size, text_length, inner_dim, generator=generator),
        audio_encoder_hidden_states=torch.randn(batch_size, text_length, audio_inner_dim, generator=generator),
        timestep=timestep,
        sigma=timestep,
        encoder_attention_mask=mask,
        audio_encoder_attention_mask=mask,
        num_frames=frames,
        height=height,
        width=width,
        audio_num_frames=audio_frames,
        use_cross_timestep=True,
        return_dict=False,
    )


def relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return ((output - reference).norm() / reference.norm()).item()


def run(rank: int, path: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    import diffusers.hooks.tensor_parallel
    from diffusers import LTX2VideoTransformer3DModel
    from torch.distributed.device_mesh import DeviceMesh

    import parallel

    # diffusers splits models on accelerators only, DTensor runs on CPU too.
    # Without an accelerator, the current one is the CPU, of which diffusers
    # needs a device count.
    diffusers.hooks.tensor_parallel._SUPPORTED_TP_DEVICES += ("cpu",)
    torch.accelerator.device_count = lambda: 1
    whole = LTX2VideoTransformer3DModel.from_pretrained(path)
    split = parallel.load_transformer(path, DeviceMesh("cpu", list(range(WORLD_SIZE))), dtype=torch.float32)
    with torch.no_grad():
        expected = whole(**inputs())
        actual = split(**inputs())
    if rank == 0:
        errors = {name: relative_error(a, e) for name, a, e in zip(("video", "audio"), actual, expected)}
        print(", ".join(f"{name}: relative error {error:.2e}" for name, error in errors.items()))
        print("PASS" if max(errors.values()) < TOLERANCE else f"FAIL: above {TOLERANCE:.0e}")
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
        mp.spawn(run, args=(path, port), nprocs=WORLD_SIZE)


if __name__ == "__main__":
    main()
