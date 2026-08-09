"""Parity gate: repo-local utilities vs the shared ``vector_core`` library.

Proves — on seeded fixtures — that swapping the duplicated preprocessing,
periodic-linear embedding, and Procrustes era-alignment utilities for
``vector_core`` is a zero-behavior-change refactor:

- ``RealMLPPreprocessor`` (the ``pipeline/realmlp_preproc.py`` version with the
  ``clip`` arg) fit/transform -> max abs diff == 0.0, float32.
- ``audit_current_scaling`` -> identical dict.
- ``load_alignment`` / ``align_batch`` (``assets/era_procrustes_align.py``) ->
  max abs diff == 0.0, float32.
- ``PLEmbedding`` (torch-gated): identical construction + forward, exact match.

The local copies are loaded by explicit file path so that once they are deleted
(the point of the migration) these comparisons ``skip`` cleanly rather than
erroring — the gate has already served its purpose before deletion.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SEED = 20260809


def _load_local(relpath: str, modname: str):
    """Import a repo-local module by file path; skip if it has been removed."""
    p = REPO / relpath
    if not p.exists():
        pytest.skip(f"local module {relpath} removed (adopted vector_core)")
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture_scaling():
    rng = np.random.default_rng(SEED)
    n, d = 200, 6
    scale = rng.uniform(0.5, 5.0, d)
    shift = rng.uniform(-3.0, 3.0, d)
    z = (rng.standard_normal((n, d)) * scale + shift).astype(np.float32)
    z[0, 0] = 50.0  # outlier so clip[-3,3] actually bites
    seasons = [str(2015 + (i % 5)) for i in range(n)]
    mask = (rng.uniform(size=(n, d)) > 0.1).astype(np.float32)
    feats = [f"f{i}" for i in range(d)]
    return z, seasons, mask, feats


def test_realmlp_preprocessor_parity():
    import vector_core as vc

    local = _load_local("pipeline/realmlp_preproc.py", "local_realmlp_preproc")
    z, seasons, mask, feats = _fixture_scaling()

    lp = local.RealMLPPreprocessor(feats, mode="robust", clip=3.0)
    vp = vc.RealMLPPreprocessor(feats, mode="robust", clip=3.0)
    lout = lp.fit_transform(z.copy(), seasons, mask.copy(), by_season=True)
    vout = vp.fit_transform(z.copy(), seasons, mask.copy(), by_season=True)

    assert lout.dtype == np.float32
    assert vout.dtype == np.float32
    assert lout.shape == vout.shape
    assert np.max(np.abs(lout - vout)) == 0.0


def test_audit_current_scaling_parity():
    import vector_core as vc

    local = _load_local("pipeline/realmlp_preproc.py", "local_realmlp_audit")
    z, seasons, mask, feats = _fixture_scaling()
    z_scaled = vc.RealMLPPreprocessor(feats, clip=3.0).fit_transform(z, seasons, mask)

    manifest = {"features": feats}
    assert local.audit_current_scaling(z_scaled, manifest) == vc.audit_current_scaling(z_scaled, manifest)


def _drift_fixture(tmp_path: Path) -> Path:
    rng = np.random.default_rng(SEED + 1)

    def rot(dim: int):
        q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        return q.astype(np.float32).tolist()

    drift = {
        "method": "orthogonal_procrustes",
        "chainedToRoot": {
            "2015": rot(6),  # full-D rotation
            "2016": rot(4),  # subset (< D) -> exercises partial-align path
        },
    }
    p = tmp_path / "drift.json"
    p.write_text(json.dumps(drift))
    return p


def test_era_alignment_parity(tmp_path):
    import vector_core as vc

    local = _load_local("assets/era_procrustes_align.py", "local_era_align")
    drift_path = _drift_fixture(tmp_path)

    # local load_alignment reads a hardcoded module path -> point it at the fixture
    local.DRIFT = drift_path
    lal = local.load_alignment()
    val = vc.load_alignment(drift_path)

    assert set(lal["chains"]) == set(val["chains"])
    for s in lal["chains"]:
        assert np.max(np.abs(lal["chains"][s] - val["chains"][s])) == 0.0

    rng = np.random.default_rng(SEED + 2)
    n, d = 60, 6
    z = rng.standard_normal((n, d)).astype(np.float32)
    seasons = [["2015", "2016", "2099"][i % 3] for i in range(n)]  # 2099 missing -> identity fallback

    lb = local.align_batch(z.copy(), seasons, lal["chains"])
    vb = vc.align_batch(z.copy(), seasons, val["chains"])

    assert lb.dtype == np.float32
    assert vb.dtype == np.float32
    assert lb.shape == vb.shape
    assert np.max(np.abs(lb - vb)) == 0.0


def test_pl_embedding_parity():
    torch = pytest.importorskip("torch")

    import vector_core as vc

    local = _load_local("pipeline/realmlp_preproc.py", "local_realmlp_ple")
    num_features, d_out, k = 6, 16, 8

    torch.manual_seed(SEED)
    lpe = local.PLEmbedding(num_features, d_out=d_out, k=k)
    torch.manual_seed(SEED)
    vpe = vc.PLEmbedding(num_features, d_out=d_out, k=k)

    for (_, lp), (_, vp) in zip(lpe.named_parameters(), vpe.named_parameters(), strict=True):
        assert torch.max(torch.abs(lp - vp)).item() == 0.0

    torch.manual_seed(SEED + 5)
    x = torch.randn(7, num_features)
    lpe.eval()
    vpe.eval()
    with torch.no_grad():
        lo = lpe(x)
        vo = vpe(x)

    assert lo.shape == vo.shape == (7, num_features, d_out)
    assert torch.max(torch.abs(lo - vo)).item() == 0.0
