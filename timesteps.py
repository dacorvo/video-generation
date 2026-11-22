"""How the LTX-2.5 transformer of diffusers runs a clip that starts from frames with two timesteps, in place of one per
video token.

Such a clip gives each video token a timestep: 0 on the tokens of the frames it starts from, the timestep of the clip on
the others. The transformer embeds each of them, 9 x 4096 values per token, 1.1 GiB per NeuronCore at 2048x1024 on 16
cores, from which each block makes more tensors of that size. The Neuron runtime fails to load the graph of such a
block (NRT_RESOURCE), while it loads the graph of a clip with one timestep.

Within `two_timesteps`, the transformer embeds only the two timesteps, and each block selects, token by token, the
modulation of the one or the other: the same values, without a tensor of 9 x 4096 values per token.
"""
from __future__ import annotations

import contextlib

import torch


class TwoTimestepEmbedding(torch.nn.Module):
    """The timestep embedding `embed` of the transformer, for a timestep per video token of two values, 0 and the
    timestep of the clip. In place of the embedding of each token, it returns, for each clip, those of the timestep of
    the clip and of 0, followed by the mask of the tokens at 0, for `two_timestep_mod_params`, and, for the output layer,
    the embedding of each token."""

    def __init__(self, embed: torch.nn.Module):
        super().__init__()
        self.embed = embed

    def forward(self, timestep, added_cond_kwargs=None, batch_size=None, hidden_dtype=None):
        if timestep.numel() == batch_size:
            return self.embed(timestep, added_cond_kwargs, batch_size, hidden_dtype)
        timestep = timestep.view(batch_size, -1)
        clip = timestep.amax(dim=1)
        # On a range of tokens all at 0, `clip` is 0 too: they all take its embedding.
        at_zero = timestep != clip[:, None]
        if not torch.all(torch.where(at_zero, timestep, 0) == 0):
            raise NotImplementedError("a timestep per video token of more than two values")
        temb, embedded = self.embed(
            torch.cat([clip, torch.zeros_like(clip)]), added_cond_kwargs, 2 * batch_size, hidden_dtype
        )
        temb = torch.cat([*temb.view(2, batch_size, -1), at_zero.to(temb.dtype)], dim=-1)
        embedded = torch.where(at_zero[..., None], embedded[batch_size:, None], embedded[:batch_size, None])
        return temb, embedded.flatten(0, 1)


def two_timestep_mod_params(get_mod_params, scale_shift_table: torch.Tensor, temb: torch.Tensor, batch_size: int):
    """`get_mod_params`, the modulation of the blocks, on the output of `TwoTimestepEmbedding`: the modulation of each
    video token, of the timestep of the clip or of 0."""
    num, dim = scale_shift_table.shape
    if temb.shape[-1] == num * dim:
        return get_mod_params(scale_shift_table, temb, batch_size)
    clip, zero, at_zero = temb.split([num * dim, num * dim, temb.shape[-1] - 2 * num * dim], dim=-1)
    table = scale_shift_table[None, None].to(temb.device)
    clip = (table + clip.reshape(batch_size, 1, num, dim)).unbind(dim=2)
    zero = (table + zero.reshape(batch_size, 1, num, dim)).unbind(dim=2)
    at_zero = at_zero.transpose(1, 2) > 0
    return tuple(torch.where(at_zero, z, c) for z, c in zip(zero, clip))


@contextlib.contextmanager
def two_timesteps(transformer):
    """Within it, `transformer` runs a timestep per video token of two values with two timesteps (see the module)."""
    block_class = type(transformer.transformer_blocks[0])
    get_mod_params, embed = block_class.get_mod_params, transformer.time_embed

    def mod_params(scale_shift_table, temb, batch_size):
        return two_timestep_mod_params(get_mod_params, scale_shift_table, temb, batch_size)

    block_class.get_mod_params = staticmethod(mod_params)
    transformer.time_embed = TwoTimestepEmbedding(embed)
    try:
        yield
    finally:
        block_class.get_mod_params = staticmethod(get_mod_params)
        transformer.time_embed = embed
