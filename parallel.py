"""Tensor parallelism for the LTX-2.5 transformer of diffusers.

diffusers splits a model over the ranks of a tensor-parallel group as it loads
it, when the model's class has a `_tp_plan`. `LTX2VideoTransformer3DModel` has
none, so `TensorParallelLTX2Transformer` adds one. It splits the query, key,
value and gate projections of every attention by columns, which leaves each
rank a slice of the heads, and the output projection by rows. It splits the
feed-forwards the same way.

The attention processor of diffusers cannot run on a slice of the heads: it
normalizes queries and keys over all heads, and reshapes them by the full head
count. `TensorParallelAttnProcessor` runs in its place.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from diffusers import LTX2VideoTransformer3DModel, TensorParallelConfig
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_ltx2 import apply_split_rotary_emb

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
    """The LTX-2.5 transformer saved at `path`, in `dtype`, split over the ranks of the 1-D device mesh `mesh`."""
    # Each rank reads only its slice of each split weight.
    transformer = TensorParallelLTX2Transformer.from_pretrained(
        path, dtype=dtype, parallel_config=TensorParallelConfig(mesh=mesh)
    )
    transformer.set_attn_processor(TensorParallelAttnProcessor(mesh.get_group()))
    return transformer
