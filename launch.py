"""The environment of the scripts that run on NeuronCores, and their launch through torchrun, one rank per core."""
from __future__ import annotations

import os
import subprocess
import sys

# The working directory of the demo, which .env sets (see setup.sh). It holds
# the venv, the weights and the NEFF cache.
WORK_DIR = os.environ["WORK_DIR"]


def set_neuron_environment(cores: int) -> None:
    """Sets the defaults of the Neuron runtime and compiler for a run on `cores` NeuronCores, before torch loads."""
    # The default NEFF cache of the pod is the mount of a public bucket: the
    # graphs of LTX-2.5 stay in the working directory.
    os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_DIR", os.path.join(WORK_DIR, "neff_cache"))
    # nki builds its kernels for LNC=1 unless told otherwise, while torch-neuronx
    # compiles for the hardware's LNC (2 on trn3): the first NKI kernel then
    # fails to compile.
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    # The first cores of the first chip. Each rank takes the core of its rank.
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", f"0-{cores - 1}")


def torchrun(script: str, cores: int, args: list[str]) -> int:
    """Runs `script` with `args` on `cores` ranks, and returns its exit code."""
    import torch

    # Otherwise torchrun gives each rank one CPU thread, which the models that
    # run on CPU then use.
    os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))
    return subprocess.call(
        [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={cores}", os.path.abspath(script), *args]
    )
