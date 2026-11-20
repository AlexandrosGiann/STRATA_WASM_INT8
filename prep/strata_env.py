"""Shared setup: import paths, the one NA3D substitution, and model loading.

The only change to the official computation is that
physicsnemo.experimental.models.strata.layers._na3d_func (NATTEN na3d) is
replaced by prep/na3d_chunked.na3d_chunked, validated against NATTEN in
prep/validate_na3d.py (NATTEN 0.21.7 cannot run fused/compiled on CPU).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATA = ROOT / "third_party" / "STRATA"
WEIGHTS = ROOT / "weights" / "strata_v1"
CKPT = WEIGHTS / "checkpoints" / "strata_v1.pth"
for p in (str(STRATA), str(ROOT / "prep" / "_stubs"), str(ROOT / "prep")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("AUX_DATA_ROOT", str(WEIGHTS))

import torch  # noqa: E402
import physicsnemo.experimental.models.strata.layers as _layers  # noqa: E402
from na3d_chunked import na3d_chunked  # noqa: E402

_layers._na3d_func = na3d_chunked


def load_model(tile: int):
    from screamcast.earth2studio_wrappers import ScreamcastModel

    model = ScreamcastModel.from_checkpoint(str(CKPT), bf16=False)
    model.eval()
    model.set_tile_size(tile)
    return model
