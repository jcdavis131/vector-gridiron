"""
Vector Gridiron MTNN — multi-tower, TransformerFusion, 32-d native + 16-d compat.

Ported from vector-hoops/pipeline/train_mtnn.py pattern + vector-pitch.

Architecture (V2):
- 10 families holistic ~160 feats total → ResidualTower cat([x*m,m])
  d_cat*2 → 96h GELU LN → 24d + skip residual + optional depth blocks
- TransformerFusion: d_model128 n_heads4 n_layers4 CLS → 32-d L2 (native)
  legacy 16-d = slice-first-16 + re-L2 (or linear proj if flag)
- Era embedding: n_seasons learned (12-d) → projected into transformer
- Procrustes Q chain alignment rotation-only orthogonal Procrustes via
  shared players ≥30, chained season→root (same as hoops).
- Heads: next-game FPTS regression MAE + SupCon archetype (k-means on F),
  position/role CE, optional MoE per-position expert gating.
- Params: ~224K-527K depending on families/hidden, FP16/INT8 quant <300KB gz,
  wrapper 17 towers [1,7]? for hoops compat but gridiron uses 10 families.

Families (gridiron):
  usage: rush_att, targets, RZ opp, etc ~16
  snaps: snap%, routes, route%, snap_share ~12
  age: AGE, YEAR_IN_LEAGUE, draft_round ~8
  weather: wind, temp, dome, precip ~10
  vegas: spread, total, implied_team_total ~8
  rest: B2B, rest_days, bye, short_week ~10
  def_vs_pos: def_vs_QB/RB/WR/TE, SOS_NET analog ~16
  form: lag FPTS 1-3, roll avg 3, std ~20
  rushing: YPC, YAC, broken, EPA ~30
  redzone: RZ targets, carries, TD ~20
  Total ~150-160 (padded to 160)

Inputs: Z_train [N,160] robust-scaled per-season median/IQR clip[-3,3]
M mask [N,160] for missing era families (cat includes mask as feature).
Output: emb L2 normalized 32-d native, 16-d legacy compat.

Usage:
  from pipeline.model import MTNN, GridironMTNN, ResidualTower, TransformerFusion

  fam_dims = {k: v for k,v in manifest['families'].items()}
  model = MTNN(fam_dims, n_seasons=30, d_emb=32)  # 32 native
  model_legacy = MTNN(fam_dims, n_seasons=30, d_emb=32, legacy_16d=True)

Export:
  emb = model.encode(xs_dict, ms_dict, season_ids)  # [B,32] or [B,16] if legacy
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Gridiron default family dims — realistic nflverse holistic totals
# If manifest on disk, these are overridden; totals intentionally sum to ~160
DEFAULT_FAM_DIMS: dict[str, int] = {
    "usage": 16,
    "snaps": 12,
    "age": 8,
    "weather": 10,
    "vegas": 8,
    "rest": 10,
    "def_vs_pos": 16,
    "form": 20,
    "rushing": 30,
    "redzone": 20,
}
# sum = 150, pad extra misc to reach 160 if needed via caller
# 10 families exactly per audit


class _ResBlock(nn.Module):
    """Same-width residual MLP block (d -> hidden -> d)"""

    def __init__(self, d: int, d_hidden: int = 96):
        super().__init__()
        self.fc1 = nn.Linear(d, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(y)))) + y)


class ResidualTower(nn.Module):
    """
    RealMLP pattern: cat([x*m, m]) per tower, then d_cat*2 → 96h GELU LN → 24d + skip.

    Args:
        d_in: feature dim for this family
        d_out: tower output dim (24 default)
        d_hidden: hidden (96 default)
        n_blocks: extra depth blocks stacked
    """

    def __init__(self, d_in: int, d_out: int = 24, d_hidden: int = 96, n_blocks: int = 1):
        super().__init__()
        d_cat = d_in * 2
        self.fc1 = nn.Linear(d_cat, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_cat, d_out) if d_cat != d_out else nn.Identity()
        self.blocks = nn.ModuleList([_ResBlock(d_out, d_hidden) for _ in range(max(0, n_blocks - 1))])

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x,m: [B,D]
        h = torch.cat([x * m, m], dim=-1)  # [B, 2*D] RealMLP masking cat([x*m,m])
        y = self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(h)))) + self.skip(h))
        for blk in self.blocks:
            y = blk(y)
        return y


class TransformerFusion(nn.Module):
    """
    v5/6: self-attention across tower tokens.

    Each tower output is a token; season token + learned [CLS] prepended.
    Pre-LN Transformer encoder (batch_first). CLS state → embedding L2.

    Gridiron bump: d_model 128 default (hoops 96), n_heads 4 n_layers 4.

    Args:
        n_towers: number of family towers
        d_tower: tower output dim (24)
        n_seasons: number of distinct seasons for embedding
        d_season: season embedding dim (12)
        d_emb: output embedding dim (32 native, 16 legacy)
        d_model: transformer width (128 gridiron v2)
        n_layers: transformer depth (4)
        n_heads: attn heads (4)
        ff: feedforward dim (256)
        dropout: transformer dropout (0.1)
    """

    def __init__(
        self,
        n_towers: int,
        d_tower: int,
        n_seasons: int,
        d_season: int = 12,
        d_emb: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tower_proj = nn.Linear(d_tower, d_model)
        self.season_emb = nn.Embedding(n_seasons, d_season)
        self.season_proj = nn.Linear(d_season, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, d_emb)

    def forward(self, tower_stack: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        b = tower_stack.size(0)
        tok = self.tower_proj(tower_stack)  # [B,T,d_model]
        s = self.season_proj(self.season_emb(season_ids)).unsqueeze(1)  # [B,1,d_model]
        cls = self.cls.expand(b, -1, -1)  # [B,1,d_model]
        x = self.encoder(torch.cat([cls, s, tok], dim=1))  # [B, 2+T, d_model]
        return F.normalize(self.out(x[:, 0]), dim=-1)  # [B,d_emb] L2


class GridironMTNN(nn.Module):
    """
    Gridiron MTNN — 10 families → 160 → TransformerFusion 128d 4-head CLS → 32-d L2.

    Native 32-d with 16-d legacy compat (slice first 16 + re-L2).
    Heads:
      - fpts_head: next-game fantasy points regression MAE (target MAE 4.268→3.8)
      - pos_head: position CE QB/RB/WR/TE (4-way)
      - archetype_head: k-means cluster 8-way (SupCon)
      - moe_gating: per-position experts (optional)

    Args:
        fam_dims: dict family->dim
        n_seasons: int seasons for season embedding
        d_tower: tower out dim 24
        d_tower_hidden: hidden 96
        d_emb: embedding dim 32 native
        legacy_16d: bool if True, forward returns 16-d slice re-L2 + sets legacy flag
        n_pos: positions 4 (QB/RB/WR/TE)
        n_archetype: archetypes 8
        d_model: transformer width 128
        n_layers: transformer depth 4
        n_heads: heads 4
        n_tower_blocks: tower depth 1
        dropout: transformer dropout 0.1
        moe_experts: per-position experts count 4 = n_pos
    """

    def __init__(
        self,
        fam_dims: dict[str, int],
        n_seasons: int,
        d_tower: int = 24,
        d_tower_hidden: int = 96,
        d_emb: int = 32,
        legacy_16d: bool = False,
        n_pos: int = 4,
        n_archetype: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_tower_blocks: int = 1,
        dropout: float = 0.1,
        moe_experts: int = 4,
        ff: int = 256,
    ):
        super().__init__()
        self.families = sorted(fam_dims)
        self.fam_dims = fam_dims
        self.n_seasons = n_seasons
        self.d_emb_native = d_emb
        self.legacy_16d = legacy_16d
        self.d_emb_eff = 16 if legacy_16d else d_emb
        self.moe_experts = moe_experts

        # towers
        self.towers = nn.ModuleDict(
            {
                fam: ResidualTower(
                    fam_dims[fam],
                    d_out=d_tower,
                    d_hidden=d_tower_hidden,
                    n_blocks=n_tower_blocks,
                )
                for fam in self.families
            }
        )
        # fusion — transformer v2 128d
        self.fusion = TransformerFusion(
            len(self.families),
            d_tower,
            n_seasons,
            d_emb=d_emb,  # native 32 always inside transformer; legacy sliced after
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff=ff,
            dropout=dropout,
        )

        # heads
        def head_fn(out_dim, hidden=64):
            return nn.Sequential(
                nn.Linear(d_emb, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, out_dim),
            )

        self.fpts_head = nn.Linear(d_emb, 1)  # next-game MAE regression
        self.pos_head = head_fn(n_pos)
        self.archetype_head = head_fn(n_archetype)
        # MoE gating per-position (optional)
        self.moe_gate = nn.Linear(d_emb, moe_experts)
        self.moe_experts_fpts = nn.ModuleList([nn.Linear(d_emb, 1) for _ in range(moe_experts)])

        # compat projection 32→16 optional (if legacy flag uses linear vs slice)
        # Simpler per task: slice first 16 dims + re-L2 (no extra params), but we keep
        # an optional linear proj for future learnable compat.
        self.legacy_proj = nn.Linear(d_emb, 16) if legacy_16d else None

    def encode(
        self, xs: dict[str, torch.Tensor], ms: dict[str, torch.Tensor], season_ids: torch.Tensor
    ) -> torch.Tensor:
        parts = torch.stack([self.towers[fam](xs[fam], ms[fam]) for fam in self.families], dim=1)  # [B,T,d_tower]
        emb_native = self.fusion(parts, season_ids)  # [B,32] L2
        if self.legacy_16d:
            # per task: slice first 16 + re-L2 (cheap, no extra training)
            # If legacy_proj exists and we want learned, you could use it — but default slice.
            if self.legacy_proj is not None:
                # Option: linear proj then L2 (more expressive) — currently unused unless caller swaps
                # Return slice for reproducibility unless explicitly using proj.
                pass
            emb = F.normalize(emb_native[:, :16], dim=-1)
            return emb
        return emb_native

    def forward(self, xs: dict[str, torch.Tensor], ms: dict[str, torch.Tensor], season_ids: torch.Tensor):
        emb = self.encode(xs, ms, season_ids)
        # For head evaluation, use native 32-d before slice when legacy;
        # recompute native if needed. Simpler: if legacy, encode still called slice in encode;
        # for heads we want 32-d behavior — we re-run fusion native via hack:
        # Actually encode returns sliced if legacy flag true, so for heads we should use native emb.
        # Fix: if legacy, call fusion separately for heads with d_emb unchanged then slice only for returned emb.
        # Quick workaround: if legacy, emb for loss is still sliced; heads trained on sliced 16-d too (okay for compat).
        # Heads dimensions mismatch: they expect 32 input, but emb is 16 if sliced.
        # So we separate logic: in forward we want emb_native for heads.

        # Recompute native if legacy (cheap)
        if self.legacy_16d:
            parts = torch.stack([self.towers[fam](xs[fam], ms[fam]) for fam in self.families], dim=1)
            emb_native = self.fusion(parts, season_ids)  # [B,32]
            emb_for_heads = emb_native
        else:
            emb_for_heads = emb

        fpts = self.fpts_head(emb_for_heads).squeeze(-1)  # [B]
        # MoE mixture
        gate = torch.softmax(self.moe_gate(emb_for_heads), dim=-1)  # [B, E]
        expert_out = torch.stack([e(emb_for_heads).squeeze(-1) for e in self.moe_experts_fpts], dim=1)  # [B,E]
        fpts_moe = (gate * expert_out).sum(-1)

        out = {
            "fpts": fpts,
            "fpts_moe": fpts_moe,
            "gate": gate,
            "pos": self.pos_head(emb_for_heads),
            "archetype": self.archetype_head(emb_for_heads),
        }
        return emb, out

    def export_16d(self, emb32: torch.Tensor) -> torch.Tensor:
        """Backward compat: 32-d → 16-d slice + L2."""
        return F.normalize(emb32[:, :16], dim=-1)


# Alias MTNN expected by scripts/export_onnx.py
class MTNN(GridironMTNN):
    """
    Hoops-compat wrapper: accepts hoops-style constructor signature.

    Hoops calls MTNN(fam_dims, n_seasons, d_tower=32, d_tower_hidden=160, d_emb=48...)
    We remap to gridiron defaults: d_tower 24 hidden 96 d_emb 32 d_model128.

    Also accepts n_game, n_skills etc ignored for compat.
    """

    def __init__(
        self,
        fam_dims: dict[str, int],
        n_seasons: int = 30,
        d_tower: int = 24,
        d_tower_hidden: int = 96,
        d_emb: int = 32,
        n_game: int = 14,
        n_skills: int = 0,
        fusion_mode: str = "transformer",
        n_tower_blocks: int = 1,
        mlp_heads: bool = False,
        d_head_hidden: int = 64,
        d_model: int = 128,
        n_fusion_layers: int = 4,
        n_attn_heads: int = 4,
        d_fusion_hidden: int | None = None,
        **kwargs,
    ):
        # kwargs may contain: legacy_16d, n_pos, n_archetype etc from gridiron calls
        legacy_16d = kwargs.pop("legacy_16d", False)
        n_pos = kwargs.pop("n_pos", 4)
        n_archetype = kwargs.pop("n_archetype", 8)
        dropout = kwargs.pop("dropout", 0.1)
        moe_experts = kwargs.pop("moe_experts", 4)
        # ignore hoops-only kwargs: d_skill_hidden, n_form, n_injury, n_bbref, etc
        super().__init__(
            fam_dims=fam_dims,
            n_seasons=n_seasons,
            d_tower=d_tower,
            d_tower_hidden=d_tower_hidden,
            d_emb=d_emb,
            legacy_16d=legacy_16d,
            n_pos=n_pos,
            n_archetype=n_archetype,
            d_model=d_model,
            n_layers=n_fusion_layers,
            n_heads=n_attn_heads,
            dropout=dropout,
            moe_experts=moe_experts,
            ff=d_fusion_hidden or 256,
        )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_report(model: nn.Module) -> dict:
    return {
        "total": count_params(model),
        "towers": sum(count_params(t) for t in model.towers.values()),
        "fusion": count_params(model.fusion),
        "heads": count_params(model.fpts_head) + count_params(model.pos_head) + count_params(model.archetype_head),
    }


if __name__ == "__main__":
    # quick smoke
    import torch

    fam_dims = DEFAULT_FAM_DIMS
    m = MTNN(fam_dims, n_seasons=30, d_emb=32)
    print(f"MTNN 32-d native params: {count_params(m)} {param_report(m)}")
    xs = {f: torch.randn(2, d) for f, d in fam_dims.items()}
    ms = {f: torch.ones(2, d) for f, d in fam_dims.items()}
    season_ids = torch.tensor([0, 29])
    emb, out = m(xs, ms, season_ids)
    print(f"emb {emb.shape} L2 {torch.norm(emb, dim=1)}")
    print(f"fpts {out['fpts'].shape} moe {out['fpts_moe'].shape}")
    # legacy
    m16 = MTNN(fam_dims, n_seasons=30, d_emb=32, legacy_16d=True)
    emb16, _ = m16(xs, ms, season_ids)
    print(f"legacy 16-d {emb16.shape}")
    print("ok")
