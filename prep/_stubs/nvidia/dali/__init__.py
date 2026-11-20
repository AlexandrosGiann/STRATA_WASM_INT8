"""Import-only stub of NVIDIA DALI (CUDA training dataloader).

screamcast/pipelines.py imports DALI at module level, but the inference path
used here never builds a DALI pipeline. Any attribute access raises.
"""


class _Missing:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, item):
        raise RuntimeError(f"nvidia.dali stub: {self._name}.{item} is not available on CPU")

    def __call__(self, *a, **k):
        raise RuntimeError(f"nvidia.dali stub: {self._name} is not available on CPU")


fn = _Missing("fn")
types = _Missing("types")
