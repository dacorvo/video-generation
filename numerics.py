"""Changes to how the LTX-2.5 transformer computes, for its precision on NeuronCores."""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from diffusers.models.activations import GELU


def _gelu(self, gate: torch.Tensor) -> torch.Tensor:
    """`F.gelu` of `gate` in fp32, rounded to its dtype."""
    return F.gelu(gate.float(), approximate=self.approximate).to(gate.dtype)


def gelu_in_fp32(model: torch.nn.Module) -> None:
    """Makes the gelus of `model` compute in fp32 and round to their input dtype, as PyTorch does on CPU."""
    # torch-neuronx computes a bf16 gelu with bf16 intermediates: where the
    # output is small, for negative inputs, the error grows to the size of the
    # output. On the feed-forwards of LTX-2.5, that doubles the error of the
    # transformer's video output.
    for module in model.modules():
        if isinstance(module, GELU):
            module.gelu = types.MethodType(_gelu, module)
