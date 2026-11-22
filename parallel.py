"""How the LTX-2.5 models of diffusers split over the ranks of a group: the
transformer by tensor parallelism, and optionally by its video tokens on top,
the decoder of the video VAE by frames.

diffusers splits a model over the ranks of a tensor-parallel group as it loads
it, when the model's class has a `_tp_plan`. `LTX2VideoTransformer3DModel` has
none, so `TensorParallelLTX2Transformer` adds one. It splits the query, key,
value and gate projections of every attention by columns, which leaves each
rank a slice of the heads, and the output projection by rows. It splits the
feed-forwards the same way.

The attention processor of diffusers cannot run on a slice of the heads: it
normalizes queries and keys over all heads, and reshapes them by the full head
count. `TensorParallelAttnProcessor` runs in its place.

With sequence parallelism on top, each rank of a sequence-parallel group runs
the transformer on a range of the video tokens, with all the audio tokens and
the text. Only two attentions need the video tokens of the other ranks: the
video self-attention, where an all-to-all swaps tokens for heads around the
attention, and the video-to-audio attention, which gathers its keys and values.

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

    def attend(self, query, key, value, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """The attention of `query` over `key` and `value`, all as (batch, tokens, heads, head dim)."""
        return dispatch_attention_fn(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

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

        hidden_states = self.attend(
            query.unflatten(2, (heads, -1)), key.unflatten(2, (heads, -1)), value.unflatten(2, (heads, -1)), attention_mask
        )
        if attn.to_gate_logits is not None:
            # The factor of 2 makes zero logits a gate of 1.
            hidden_states = hidden_states * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class SequenceParallelSelfAttnProcessor(TensorParallelAttnProcessor):
    """The video self-attention of `TensorParallelAttnProcessor`, the video tokens split over the ranks of `sp_group`.

    An all-to-all swaps the tokens of this rank for the heads of this rank of the group, so that the rank attends over
    all tokens with its heads, and a second all-to-all swaps them back.
    """

    def __init__(self, group: dist.ProcessGroup, sp_group: dist.ProcessGroup):
        super().__init__(group)
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)

    def tokens_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """`x`, the tokens of this rank with all heads, as all tokens with the heads of this rank of the group."""
        batch, tokens, heads, dim = x.shape
        # The heads of SP rank i go to it, and the tokens of SP rank i come back in place i.
        chunks = x.unflatten(2, (self.sp_size, heads // self.sp_size)).permute(2, 0, 1, 3, 4).contiguous()
        received = funcol.all_to_all_single(chunks, None, None, group=self.sp_group)
        return received.permute(1, 0, 2, 3, 4).reshape(batch, self.sp_size * tokens, heads // self.sp_size, dim)

    def heads_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """`x`, all tokens with the heads of this rank of the group, as the tokens of this rank with all heads."""
        batch, tokens, heads, dim = x.shape
        chunks = x.unflatten(1, (self.sp_size, tokens // self.sp_size)).permute(1, 0, 2, 3, 4).contiguous()
        received = funcol.all_to_all_single(chunks, None, None, group=self.sp_group)
        return received.permute(1, 2, 0, 3, 4).reshape(batch, tokens // self.sp_size, self.sp_size * heads, dim)

    def attend(self, query, key, value, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("a video self-attention mask under sequence parallelism")
        query, key, value = (self.tokens_to_heads(x) for x in (query, key, value))
        return self.heads_to_tokens(super().attend(query, key, value, attention_mask))


class SequenceParallelKeysAttnProcessor(TensorParallelAttnProcessor):
    """The video-to-audio attention of `TensorParallelAttnProcessor`, the video tokens, its keys and values, split over
    the ranks of `sp_group`: it gathers them from all ranks of the group."""

    def __init__(self, group: dist.ProcessGroup, sp_group: dist.ProcessGroup):
        super().__init__(group)
        self.sp_group = sp_group

    def attend(self, query, key, value, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("a video-to-audio attention mask under sequence parallelism")
        key, value = (funcol.all_gather_tensor(x.contiguous(), gather_dim=1, group=self.sp_group) for x in (key, value))
        return super().attend(query, key, value, attention_mask)


def load_transformer(
    path: str, mesh, sp_group: dist.ProcessGroup | None = None, dtype: torch.dtype = torch.bfloat16
) -> TensorParallelLTX2Transformer:
    """The LTX-2.5 transformer saved at `path`, in `dtype`, split over the ranks of the 1-D device mesh `mesh`, its
    gelus computed in fp32. With `sp_group`, it runs on the range of the video tokens of this rank of that group too
    (see `split_video_tokens`)."""
    # Each rank reads only its slice of each split weight.
    transformer = TensorParallelLTX2Transformer.from_pretrained(
        path, dtype=dtype, parallel_config=TensorParallelConfig(mesh=mesh)
    )
    transformer.set_attn_processor(TensorParallelAttnProcessor(mesh.get_group()))
    if sp_group is not None:
        # The other attentions take the video tokens of this rank, the audio tokens or the text as queries, and the
        # audio tokens or the text as keys, which every rank holds.
        self_attention = SequenceParallelSelfAttnProcessor(mesh.get_group(), sp_group)
        video_keys = SequenceParallelKeysAttnProcessor(mesh.get_group(), sp_group)
        for block in transformer.transformer_blocks:
            block.attn1.set_processor(self_attention)
            block.video_to_audio_attn.set_processor(video_keys)
    numerics.gelu_in_fp32(transformer)
    return transformer


def split_video_tokens(kwargs: dict, sp_group: dist.ProcessGroup) -> dict:
    """The arguments of a call of the transformer, its video tokens, their positions and, if it has one per token, their
    timesteps, cut to the range of this rank of `sp_group`.

    A clip that starts from frames has a timestep per video token, 0 on the tokens of those frames. The transformer
    embeds it per token, and the audio tokens take their own timestep, so each rank only needs those of its tokens.
    """
    for name in ("video_self_attention_mask", "video_keyframes_mask", "perturbation_mask"):
        if kwargs.get(name) is not None:
            raise NotImplementedError(f"{name} under sequence parallelism")
    if kwargs.get("video_coords") is None:
        raise NotImplementedError("sequence parallelism without the video positions")
    rank, size = dist.get_rank(sp_group), dist.get_world_size(sp_group)
    tokens = kwargs["hidden_states"].shape[1]
    if tokens % size:
        raise ValueError(f"{tokens} video tokens do not split over {size} ranks")
    start, end = rank * tokens // size, (rank + 1) * tokens // size
    # The latents of LTX2Pipeline are a transposed view: torch-neuronx takes seconds to copy a slice of them to the
    # device, and milliseconds once the slice is contiguous.
    split = {"hidden_states": kwargs["hidden_states"][:, start:end].contiguous(),
             "video_coords": kwargs["video_coords"][:, :, start:end].contiguous()}
    if kwargs["timestep"].ndim > 1:
        split["timestep"] = kwargs["timestep"][:, start:end].contiguous()
    return {**kwargs, **split}


def gather_video_tokens(video: torch.Tensor, sp_group: dist.ProcessGroup) -> torch.Tensor:
    """The video output of the transformer for all tokens, from the range of this rank of `sp_group`."""
    return funcol.all_gather_tensor(video.contiguous(), gather_dim=1, group=sp_group)


def frame_ranges(num_frames: int, world_size: int) -> list[tuple[int, int]]:
    """The [start, end) latent frames of each rank, as even as possible, the longer ranges first."""
    size, extra = divmod(num_frames, world_size)
    starts = [rank * size + min(rank, extra) for rank in range(world_size + 1)]
    return list(zip(starts[:-1], starts[1:]))


@torch.compiler.disable
def _pad_with_neighbors(
    hidden_states: torch.Tensor, group: dist.ProcessGroup, rank: int, world_size: int
) -> torch.Tensor:
    """`hidden_states`, the frames of this rank of `group`, padded in time with the last frame of the previous rank and
    the first frame of the next one, or with its own at the edges of the clip.

    It runs between the compiled graphs of the decoder. Inside them, the all-gather of ranks whose frames differ in
    number compiles to collectives that the Neuron runtime refuses to match across chips.
    """
    first, last = hidden_states[:, :, :1], hidden_states[:, :, -1:]
    # The first and last frames of each rank, in rank order.
    edges = funcol.all_gather_tensor(torch.cat([first, last], dim=2).contiguous(), gather_dim=2, group=group)
    left = edges[:, :, 2 * rank - 1 : 2 * rank] if rank > 0 else first
    right = edges[:, :, 2 * rank + 2 : 2 * rank + 3] if rank < world_size - 1 else last
    return torch.cat([left, hidden_states, right], dim=2)


def _frame_split_conv_forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """`LTX2VideoCausalConv3d.forward`, its time padding taken from the neighboring ranks when it is not causal."""
    if causal:
        return LTX2VideoCausalConv3d.forward(self, hidden_states, causal=causal)
    return self.conv(_pad_with_neighbors(hidden_states, *self.frame_split))


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
