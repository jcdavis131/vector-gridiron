"""
RealMLP Robust Preprocessing for Vector Gridiron — pipeline version

Matches assets/realmlp_preproc.py but importable from pipeline.

Implements:
- RobustScaler median / IQR per-season with clip [-3,3]
- PLEmbedding periodic sin/cos k=8 d_out=16 proj linear 2k→16
- RealMLPPreprocessor top-level with by_season fit, global fallback

Reference: RealMLP Holzmüller et al 2024
Same as hoops per-100 z stability but for NFL MAE 4.268 → 3.8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RobustScaler:
    """Median / IQR scaling with hard clip [-clip, clip] (RealMLP robust scaling)."""

    def __init__(self, clip: float = 3.0, eps: float = 1e-6):
        self.clip = clip
        self.eps = eps
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, Z: np.ndarray, mask: np.ndarray | None = None):
        """
        Z: [N, D] float32
        mask: [N, D] 0/1 valid indicator (era-missing families)
        """
        if mask is None:
            mask = np.ones_like(Z)
        D = Z.shape[1]
        medians = np.zeros(D, dtype=np.float32)
        iqrs = np.ones(D, dtype=np.float32)
        for d in range(D):
            vals = Z[mask[:, d] > 0, d]
            if len(vals) < 10:
                continue
            medians[d] = np.median(vals)
            q75, q25 = np.percentile(vals, [75, 25])
            iqrs[d] = max(float(q75 - q25), 1e-6)
        self.median_ = medians
        self.iqr_ = iqrs
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            raise ValueError("fit first")
        Z_scaled = (Z - self.median_) / (self.iqr_ + self.eps)
        Z_scaled = np.clip(Z_scaled, -self.clip, self.clip)
        return Z_scaled.astype(np.float32)

    def fit_transform(self, Z: np.ndarray, mask: np.ndarray | None = None):
        return self.fit(Z, mask).transform(Z)


class PLEmbedding(nn.Module):
    """
    Periodic Linear embeddings (RealMLP / FT-Transformer):
    For each scalar feature x, produce [sin(2π k x), cos(2π k x)] * linear
    k = learnable frequencies per feature.

    Input: [B, F] scalars
    Output: [B, F, d_out]
    """

    def __init__(self, num_features: int, d_out: int = 16, k: int = 8):
        super().__init__()
        self.num_features = num_features
        self.d_out = d_out
        self.k = k
        # learnable frequencies per feature per k — small init 0.1
        self.freq = nn.Parameter(torch.randn(num_features, k) * 0.1)
        self.proj = nn.Linear(2 * k, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        freq = self.freq.unsqueeze(0)  # [1, F, k]
        x_exp = x.unsqueeze(-1) * freq  # [B, F, k]
        sin_emb = torch.sin(2 * np.pi * x_exp)
        cos_emb = torch.cos(2 * np.pi * x_exp)
        periodic = torch.cat([sin_emb, cos_emb], dim=-1)  # [B, F, 2k]
        return self.proj(periodic)  # [B, F, d_out]


class RealMLPPreprocessor:
    """
    Top-level wrapper that stores per-season RobustScaler.
    Honest: fit on train seasons only, global fallback for unseen.

    Usage:
      preproc = RealMLPPreprocessor(feature_names)
      preproc.fit(Z_train, seasons_train, mask_train, by_season=True)
      Z_val = preproc.transform(Z_val, seasons_val)
    """

    def __init__(self, feature_names: list[str], mode: str = "robust", clip: float = 3.0):
        self.feature_names = feature_names
        self.mode = mode
        self.clip = clip
        self.scalers: dict[str, RobustScaler] = {}
        self.global_scaler = RobustScaler(clip=clip)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path):
        manifest = json.loads(Path(manifest_path).read_text())
        return cls(manifest["features"])

    def fit(
        self,
        Z: np.ndarray,
        seasons: list[str],
        mask: np.ndarray | None = None,
        by_season: bool = True,
    ):
        """Fit scaler per season (era-honest) or globally."""
        if by_season:
            from collections import defaultdict

            by_s = defaultdict(list)
            for i, s in enumerate(seasons):
                by_s[str(s)].append(i)
            for season, idx in by_s.items():
                scaler = RobustScaler(clip=self.clip)
                Z_s = Z[idx]
                M_s = mask[idx] if mask is not None else None
                scaler.fit(Z_s, M_s)
                self.scalers[season] = scaler
        self.global_scaler.fit(Z, mask)
        return self

    def transform(self, Z: np.ndarray, seasons: list[str] | None = None) -> np.ndarray:
        if seasons is None:
            return self.global_scaler.transform(Z)
        Z_out = np.zeros_like(Z, dtype=np.float32)
        for i, season in enumerate(seasons):
            scaler = self.scalers.get(str(season), self.global_scaler)
            row = (Z[i] - scaler.median_) / (scaler.iqr_ + scaler.eps)
            Z_out[i] = row
        return np.clip(Z_out, -self.clip, self.clip).astype(np.float32)

    def fit_transform(
        self,
        Z: np.ndarray,
        seasons: list[str],
        mask: np.ndarray | None = None,
        by_season: bool = True,
    ):
        return self.fit(Z, seasons, mask, by_season=by_season).transform(Z, seasons)

    def save(self, path: str | Path):
        path = Path(path)
        payload = {
            "feature_names": self.feature_names,
            "clip": self.clip,
            "mode": self.mode,
            "scalers": {
                season: {
                    "median": scaler.median_.tolist(),
                    "iqr": scaler.iqr_.tolist(),
                }
                for season, scaler in self.scalers.items()
            },
            "global": {
                "median": self.global_scaler.median_.tolist(),
                "iqr": self.global_scaler.iqr_.tolist(),
            },
        }
        path.write_text(json.dumps(payload))
        return path

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        obj = cls(data["feature_names"], mode=data.get("mode", "robust"), clip=data.get("clip", 3.0))
        for season, vals in data["scalers"].items():
            s = RobustScaler(clip=data.get("clip", 3.0))
            s.median_ = np.array(vals["median"], dtype=np.float32)
            s.iqr_ = np.array(vals["iqr"], dtype=np.float32)
            obj.scalers[season] = s
        g = RobustScaler(clip=data.get("clip", 3.0))
        g.median_ = np.array(data["global"]["median"], dtype=np.float32)
        g.iqr_ = np.array(data["global"]["iqr"], dtype=np.float32)
        obj.global_scaler = g
        return obj


def audit_current_scaling(Z: np.ndarray, manifest: dict) -> dict:
    """Audit current per-100 z vs robust: outlier rate."""
    outlier = (np.abs(Z) > 3).mean(axis=0)
    top = np.argsort(-outlier)[:10]
    features = manifest.get("features", [f"f{i}" for i in range(Z.shape[1])])
    return {
        "mean_abs_z": float(np.abs(Z).mean()),
        "outlier_rate_gt3": float((np.abs(Z) > 3).mean()),
        "outlier_rate_gt4": float((np.abs(Z) > 4).mean()),
        "worst_features": [
            {"feature": features[i] if i < len(features) else f"f{i}", "outlier_gt3": float(outlier[i])} for i in top
        ],
    }


if __name__ == "__main__":
    print("RealMLP preprocessor — pipeline importable, run via train_mtnn.py")
    print("PL embeddings d_out=16 k=8 periodic sin/cos learnable freq")
