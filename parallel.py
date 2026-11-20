"""How the LTX-2.5 models of diffusers split over the ranks of a group: the
transformer by tensor parallelism, the decoder of the video VAE by frames.

diffusers splits a model over the ranks of a tensor-parallel group as it loads
it, when the model's class has a `_tp_plan`. `LTX2VideoTransformer3DModel` has
none, so `TensorParallelLTX2Transformer` adds one. It splits the query, key,
value and gate projections of every attention by columns, which leaves each
rank a slice of the heads, and the output projection by rows. It splits the
feed-forwards the same way.

The attention processor of diffusers cannot run on a slice of the heads: it
normalizes queries and keys over all heads, and reshapes them by the full head
count. `TensorParallelAttnProcessor` runs in its place.

With the decoder split by frames, each rank decodes a contiguous range of the
latent frames. The decoder's temporal convolutions (kernel 3, stride 1) pad
time with one frame on each side, which they repeat from the edges of the
clip: a rank takes those frames from its neighbors instead, which all ranks
exchange by an all-gather of their first and last frames, and only the first
and last ranks repeat their own. Its temporal upsamplers drop the first frame
they make, which only the first rank does. Each rank then decodes exactly its
frames of the whole decode.
"""
from __future__ import annotations

import types

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from diffusers import LTX2VideoTransformer3DModel, TensorParallelConfig
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoCausalConv3d, LTX2VideoUpsampler3d
from diffusers.models.transformers.transformer_ltx2 import apply_split_rotary_emb

import numerics

# The attentions and feed-forwards of each transformer block.
ATTENTIONS = ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")
FEED_FORWARDS = ("ff", "audio_ff")


class TensorParallelLTX2Transformer(LTX2VideoTransformer3DModel):
    """The LTX-2.5 transformer, which diffusers can split over a tensor-parallel group."""

    _tp_plan = {
        **{
            f"transformer_blocks.*.{attn}.{proj}": "colwise"
            for attn in ATTENTIONS
            for proj in ("to_q", "to_k", "to_v", "to_gate_logits")
        },
        **{f"transformer_blocks.*.{attn}.to_out.0": "rowwise" for attn in ATTENTIONS},
        **{f"transformer_blocks.*.{ff}.net.0.proj": "colwise" for ff in FEED_FORWARDS},
        **{f"transformer_blocks.*.{ff}.net.2": "rowwise" for ff in FEED_FORWARDS},
    }


class TensorParallelAttnProcessor:
    """The attention of `LTX2PerturbedAttnProcessor` without spatio-temporal guidance, on the heads of one rank of a
    tensor-parallel group."""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

    def norm(self, x: torch.Tensor, norm: torch.nn.RMSNorm) -> torch.Tensor:
        """`x`, the heads of this rank, normalized by `norm`, an RMSNorm over the heads of all ranks."""
        x_float = x.float()
        sum_sq = funcol.all_reduce(x_float.pow(2).sum(-1, keepdim=True), "sum", self.group)
        mean_sq = sum_sq / (x.shape[-1] * self.world_size)
        weight = norm.weight.chunk(self.world_size)[self.rank]
        # torch.nn.RMSNorm applies its weight before it rounds to the input dtype.
        return (x_float * torch.rsqrt(mean_sq + norm.eps) * weight.float()).to(x.dtype)

    def rope(self, x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """`x`, the heads of this rank, rotated by `freqs`, the split RoPE frequencies of all heads."""
        cos, sin = (f.chunk(self.world_size, dim=1)[self.rank] for f in freqs)
        return apply_split_rotary_emb(x, (cos, sin))

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool | None = None,
    ) -> torch.Tensor:
        # Interleaved RoPE frequencies are not laid out by head: the slice of
        # `rope` would take the wrong ones.
        if attn.rope_type != "split":
            raise NotImplementedError(f"RoPE of type {attn.rope_type!r} under tensor parallelism")
        # The blocks pass a perturbation mask to the self-attentions that
        # spatio-temporal guidance perturbs. The distilled model runs without it.
        if perturbation_mask is not None:
            raise NotImplementedError("spatio-temporal guidance under tensor parallelism")
        heads = attn.heads // self.world_size
        batch_size = hidden_states.shape[0]

        if attention_mask is not None:
            # The same mask for every head: diffusers repeats it for each, broadcasting does here.
            attention_mask = attention_mask.view(batch_size, 1, -1, attention_mask.shape[-1])
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)

        query = self.norm(attn.to_q(hidden_states), attn.norm_q)
        key = self.norm(attn.to_k(encoder_hidden_states), attn.norm_k)
        value = attn.to_v(encoder_hidden_states)
        if query_rotary_emb is not None:
            query = self.rope(query, query_rotary_emb)
            key = self.rope(key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

        hidden_states = dispatch_attention_fn(
            query.unflatten(2, (heads, -1)),
            key.unflatten(2, (heads, -1)),
            value.unflatten(2, (heads, -1)),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        if attn.to_gate_logits is not None:
            # The factor of 2 makes zero logits a gate of 1.
            hidden_states = hidden_states * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def load_transformer(path: str, mesh, dtype: torch.dtype = torch.bfloat16) -> TensorParallelLTX2Transformer:
    """The LTX-2.5 transformer saved at `path`, in `dtype`, split over the ranks of the 1-D device mesh `mesh`, its
    gelus computed in fp32."""
    # Each rank reads only its slice of each split weight.
    transformer = TensorParallelLTX2Transformer.from_pretrained(
        path, dtype=dtype, parallel_config=TensorParallelConfig(mesh=mesh)
    )
    transformer.set_attn_processor(TensorParallelAttnProcessor(mesh.get_group()))
    numerics.gelu_in_fp32(transformer)
    return transformer


def frame_ranges(num_frames: int, world_size: int) -> list[tuple[int, int]]:
    """The [start, end) latent frames of each rank, as even as possible, the longer ranges first."""
    size, extra = divmod(num_frames, world_size)
    starts = [rank * size + min(rank, extra) for rank in range(world_size + 1)]
    return list(zip(starts[:-1], starts[1:]))


def _frame_split_conv_forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """`LTX2VideoCausalConv3d.forward`, its time padding taken from the neighboring ranks when it is not causal."""
    if causal:
        return LTX2VideoCausalConv3d.forward(self, hidden_states, causal=causal)
    group, rank, world_size = self.frame_split
    first, last = hidden_states[:, :, :1], hidden_states[:, :, -1:]
    # The first and last frames of each rank, in rank order.
    edges = funcol.all_gather_tensor(torch.cat([first, last], dim=2).contiguous(), gather_dim=2, group=group)
    left = edges[:, :, 2 * rank - 1 : 2 * rank] if rank > 0 else first
    right = edges[:, :, 2 * rank + 2 : 2 * rank + 3] if rank < world_size - 1 else last
    return self.conv(torch.cat([left, hidden_states, right], dim=2))


def _frame_split_upsampler_forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """`LTX2VideoUpsampler3d.forward`, which drops the first frames it makes on the first rank only."""
    _, rank, _ = self.frame_split
    drop = self.stride[0] - 1 if rank == 0 else 0
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    if self.residual:
        residual = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
        residual = residual.repeat(1, repeats, 1, 1, 1)[:, :, drop:]
    hidden_states = self.conv(hidden_states, causal=causal)
    hidden_states = hidden_states.reshape(
        batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
    )
    hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    hidden_states = hidden_states[:, :, drop:]
    if self.residual:
        hidden_states = hidden_states + residual
    return hidden_states


def split_by_frames(vae, group: dist.ProcessGroup) -> None:
    """Makes the video VAE `vae` decode, from the latents of a whole clip, the frames of this rank of `group`."""
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    for module in vae.decoder.modules():
        if isinstance(module, LTX2VideoCausalConv3d):
            if module.kernel_size[0] != 3 or module.conv.stride[0] != 1 or module.conv.dilation[0] != 1:
                raise NotImplementedError(f"time kernel {module.kernel_size[0]}, stride {module.conv.stride[0]}")
            module.frame_split = (group, rank, world_size)
            module.forward = types.MethodType(_frame_split_conv_forward, module)
        elif isinstance(module, LTX2VideoUpsampler3d):
            module.frame_split = (group, rank, world_size)
            module.forward = types.MethodType(_frame_split_upsampler_forward, module)
    decode = vae.decode

    def decode_frames(latents: torch.Tensor, *args, **kwargs):
        start, end = frame_ranges(latents.shape[2], world_size)[rank]
        return decode(latents[:, :, start:end], *args, **kwargs)

    vae.decode = decode_frames
