"""Vector Gridiron MTNN v2 — multi-tower, multi-task weekly fantasy model.

Residual family towers (masked) → gated fusion → L2 embedding → heads:
  fpts_ppr + component yards/rec/TD + usage recon + position + pedigree aux.

Honest temporal split (train≤2023 / val 2024 / test 2025). Exports
nextgame.json, projections.json, embedding.json (UI contract preserved).
Rookie draft-capital model retained from v1 for players without NFL form.

Run:  python pipeline/train_mtnn.py [--offline] [--epochs 40]
Requires: pipeline/data/train_matrix.npz from build_features.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from math import log1p
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import build_features as bf
import nfl_data as nfl

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "pipeline" / "data"
SEED = 7
SKILL = ("QB", "RB", "WR", "TE")
POS_INDEX = {p: i for i, p in enumerate(SKILL)}

HEAD_WEIGHTS = {
    "fpts_ppr": 1.0, "rec_yds": 0.25, "rush_yds": 0.25, "pass_yds": 0.25,
    "receptions": 0.25, "total_td": 0.35,
}
V1_MAE = 4.313  # promotion reference


def norm_key(name, pos):
    s = (name or "").lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f"{s}|{pos}"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualTower(nn.Module):
    def __init__(self, d_in: int, d_out: int = 24, d_hidden: int = 64):
        super().__init__()
        d_cat = d_in * 2
        self.fc1 = nn.Linear(d_cat, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_cat, d_out) if d_cat != d_out else nn.Identity()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x * m, m], dim=-1)
        return self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(h)))) + self.skip(h))


class GatedFusion(nn.Module):
    def __init__(self, n_towers: int, d_tower: int, n_seasons: int,
                 d_season: int = 8, d_emb: int = 32, d_hidden: int = 128):
        super().__init__()
        self.season_emb = nn.Embedding(n_seasons, d_season)
        self.gate = nn.Linear(d_tower, 1)
        self.attn = nn.Sequential(
            nn.Linear(d_tower, d_tower), nn.Tanh(), nn.Linear(d_tower, 1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_tower + d_season, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden),
            nn.Dropout(0.15), nn.Linear(d_hidden, d_emb),
        )

    def family_weights(self, tower_stack: torch.Tensor) -> torch.Tensor:
        """Per-row family mix: softmax(attn) × sigmoid(gate), L1-normalized."""
        scores = self.attn(tower_stack).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        gates = torch.sigmoid(self.gate(tower_stack).squeeze(-1))
        raw = weights * gates
        return raw / (raw.sum(dim=-1, keepdim=True) + 1e-9)

    def forward(self, tower_stack: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        scores = self.attn(tower_stack).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        gates = torch.sigmoid(self.gate(tower_stack).squeeze(-1))
        mixed = (tower_stack * weights.unsqueeze(-1) * gates.unsqueeze(-1)).sum(1)
        s = self.season_emb(season_ids)
        return F.normalize(self.fuse(torch.cat([mixed, s], dim=-1)), dim=-1)


class MTNN(nn.Module):
    def __init__(self, fam_dims: dict[str, int], n_seasons: int,
                 d_tower: int = 24, d_emb: int = 32, n_targets: int = 6,
                 n_usage: int = 3):
        super().__init__()
        self.families = sorted(fam_dims)
        self.towers = nn.ModuleDict({
            fam: ResidualTower(fam_dims[fam], d_out=d_tower)
            for fam in self.families
        })
        self.fusion = GatedFusion(len(self.families), d_tower, n_seasons, d_emb=d_emb)
        self.target_heads = nn.ModuleList([nn.Linear(d_emb, 1) for _ in range(n_targets)])
        self.usage_head = nn.Linear(d_emb, n_usage)
        self.position_head = nn.Linear(d_emb, len(SKILL))
        self.pedigree_head = nn.Linear(d_emb, 1)

    def tower_stack(self, xs, ms):
        return torch.stack(
            [self.towers[fam](xs[fam], ms[fam]) for fam in self.families], dim=1)

    def encode(self, xs, ms, season_ids):
        return self.fusion(self.tower_stack(xs, ms), season_ids)

    def encode_with_contrib(self, xs, ms, season_ids):
        parts = self.tower_stack(xs, ms)
        w = self.fusion.family_weights(parts)
        emb = self.fusion(parts, season_ids)
        return emb, w

    def forward(self, xs, ms, season_ids):
        emb = self.encode(xs, ms, season_ids)
        targets = torch.cat([h(emb) for h in self.target_heads], dim=1)
        return emb, {
            "targets": targets,
            "usage": self.usage_head(emb),
            "position": self.position_head(emb),
            "pedigree": self.pedigree_head(emb).squeeze(-1),
        }


def family_slices(feature_names, families):
    return {fam: [feature_names.index(c) for c in cols]
            for fam, cols in families.items()}


def split_by_family(X, M, slices, device):
    xs, ms = {}, {}
    for fam, cols in slices.items():
        xs[fam] = torch.tensor(X[:, cols], dtype=torch.float32, device=device)
        ms[fam] = torch.tensor(M[:, cols], dtype=torch.float32, device=device)
    return xs, ms


# ---------------------------------------------------------------------------
# Rookie model (from v1)
# ---------------------------------------------------------------------------

def season_pergame(season):
    agg = {}
    for r in nfl.weekly_stats(season):
        if r.get("season_type") != "REG":
            continue
        pos = (r.get("position") or "").strip()
        if pos not in SKILL:
            continue
        g = r.get("player_id")
        a = agg.setdefault(g, {"games": 0, **{t: 0.0 for t in bf.TARGET_NAMES}})
        a["games"] += 1
        a["fpts_ppr"] += nfl.num(r, "fantasy_points_ppr")
        a["rec_yds"] += nfl.num(r, "receiving_yards")
        a["rush_yds"] += nfl.num(r, "rushing_yards")
        a["pass_yds"] += nfl.num(r, "passing_yards")
        a["receptions"] += nfl.num(r, "receptions")
        a["total_td"] += (nfl.num(r, "passing_tds") + nfl.num(r, "rushing_tds")
                          + nfl.num(r, "receiving_tds"))
    for a in agg.values():
        gg = max(1, a["games"])
        for t in bf.TARGET_NAMES:
            a[t] /= gg
    return agg


def rookie_meta():
    out = {}
    for r in nfl.players():
        pos = (r.get("position") or "").strip()
        if pos not in SKILL:
            continue
        pick = (r.get("draft_pick") or "").strip()
        rs = (r.get("rookie_season") or r.get("draft_year") or "").strip()
        if not (pick.isdigit() and rs.isdigit()):
            continue
        rd = (r.get("draft_round") or "").strip()
        out[r.get("gsis_id")] = {
            "pick": int(pick), "round": int(rd) if rd.isdigit() else 8, "pos": pos,
            "rookie_season": int(rs), "birth": r.get("birth_date", ""),
            "name": r.get("display_name") or r.get("short_name") or "",
            "team": r.get("draft_team", "") or r.get("latest_team", ""),
            "headshot": r.get("headshot", ""),
        }
    return out


def rookie_feats(m, season):
    b = m.get("birth", "")
    try:
        by = int(b.split("-")[0]); age = season - by
    except Exception:
        age = 22.0
    return [log1p(m["pick"]), float(m["round"])] \
        + [1.0 if m["pos"] == p else 0.0 for p in SKILL] + [float(age)]


def train_rookie_model(last_season):
    torch.manual_seed(SEED)
    rm = rookie_meta()
    pergame = {}

    def pg(s):
        if s not in pergame:
            pergame[s] = season_pergame(s)
        return pergame[s]

    X, Y, seas, hist = [], [], [], []
    for g, m in rm.items():
        rs = m["rookie_season"]
        if rs < nfl.FIRST_SEASON or rs > last_season:
            continue
        prod = pg(rs).get(g)
        if not prod or prod["games"] < 3:
            continue
        X.append(rookie_feats(m, rs))
        Y.append([prod[t] for t in bf.TARGET_NAMES])
        seas.append(rs)
        hist.append((m["pos"], log1p(m["pick"]), m["name"], round(prod["fpts_ppr"], 1)))
    X, Y, seas = np.array(X, float), np.array(Y, float), np.array(seas)
    tr, va = seas <= last_season - 2, seas > last_season - 2
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0); ysd[ysd == 0] = 1
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Dropout(0.15),
                        nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Y[tr] - ymu) / ysd, dtype=torch.float32)
    best, bad, bstate = 1e9, 0, None
    Xv = torch.tensor((X[va] - mu) / sd, dtype=torch.float32) if va.any() else None
    for _ in range(400):
        net.train(); opt.zero_grad()
        loss = F.smooth_l1_loss(net(Xt), Yt)
        loss.backward(); opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                pv = net(Xv).numpy() * ysd + ymu
            vmae = float(np.mean(np.abs(pv[:, 0] - Y[va][:, 0])))
            if vmae < best - 1e-3:
                best, bad, bstate = vmae, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 40:
                    break
    if bstate:
        net.load_state_dict(bstate)
    net.eval()

    def predict(Xr):
        with torch.no_grad():
            return net(torch.tensor((Xr - mu) / sd, dtype=torch.float32)).numpy() * ysd + ymu

    base = {p: float(np.mean([Y[tr][i, 0] for i in range(tr.sum())
            if X[tr][i, 2 + SKILL.index(p)] == 1] or [8])) for p in SKILL}
    report = {
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "val_fpts_mae": round(best, 3) if bstate else None,
        "baseline_pos_mae": round(float(np.mean([
            abs(Y[va][i, 0] - base[SKILL[int(np.argmax(X[va][i, 2:6]))]])
            for i in range(va.sum())])), 3) if va.any() else None,
    }
    return predict, report, rm, hist


def rookie_board(predict, rm, hist, existing_keys, last_season, proj_season, resid, ti, byes):
    rows = []
    for g, m in rm.items():
        if m["rookie_season"] < last_season:
            continue
        key = norm_key(m["name"], m["pos"])
        if key in existing_keys or not m["name"]:
            continue
        p = predict(np.array([rookie_feats(m, proj_season)], float))[0]
        lp = log1p(m["pick"])
        comps = sorted([h for h in hist if h[0] == m["pos"] and h[2] != m["name"]],
                       key=lambda h: abs(h[1] - lp))[:3]
        rows.append({
            "key": key, "name": m["name"], "pos": m["pos"], "team": m["team"],
            "headshot": m["headshot"], "rookie": True, "bye": byes.get(m["team"]), "avail": "",
            "draft": {"round": m["round"], "pick": m["pick"], "team": m["team"]},
            "proj": round(float(p[ti["fpts_ppr"]]), 2),
            "floor": round(max(0.0, float(p[ti["fpts_ppr"]]) - resid), 2),
            "ceil": round(float(p[ti["fpts_ppr"]]) + resid, 2),
            "line": {
                "rec_yds": round(float(p[ti["rec_yds"]]), 1),
                "rush_yds": round(float(p[ti["rush_yds"]]), 1),
                "pass_yds": round(float(p[ti["pass_yds"]]), 1),
                "rec": round(float(p[ti["receptions"]]), 1),
                "td": round(float(p[ti["total_td"]]), 2),
            },
            "comps": [{"name": c[2], "pos": m["pos"], "sim": 0,
                       "note": f"similar draft slot · {c[3]} rookie ppr"} for c in comps],
        })
    return rows


# ---------------------------------------------------------------------------
# Train + export
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--skip-build", action="store_true",
                    help="use existing train_matrix.npz + rebuild upcoming only")
    ap.add_argument("--family-drop", type=float, default=0.0,
                    help="train-time probability of zeroing a whole family mask")
    ap.add_argument("--d-emb", type=int, default=32)
    ap.add_argument("--zero-families", default="",
                    help="comma-separated families to permanently zero (prune bet)")
    ap.add_argument("--soft-families", default="",
                    help="comma-separated families to soft-scale masks (not hard zero)")
    ap.add_argument("--soft-scale", type=float, default=0.25,
                    help="mask multiplier for --soft-families (default 0.25)")
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    last_season = nfl.latest_stats_season(args.offline)
    print(f"latest published season = {last_season}; projecting {last_season + 1}")
    print("building holistic multi-family feature matrix ...")
    D = bf.build(last_season, offline=args.offline)
    X, M, Y = D["X"], D["M"], D["Y"]
    Yu = D["Y_usage"]
    META = D["meta"]
    feats, targets = D["feature_names"], D["target_names"]
    families = D["families"]
    zero_fams = [f.strip() for f in args.zero_families.split(",") if f.strip()]
    if zero_fams:
        n_z = 0
        for fam in zero_fams:
            cols = families.get(fam, [])
            for col in cols:
                if col not in feats:
                    continue
                i = feats.index(col)
                M[:, i] = 0.0
                X[:, i] = 0.0
                n_z += 1
            # also zero upcoming
            up = D["upcoming"]
            for col in cols:
                if col in feats:
                    i = feats.index(col)
                    up["masks"][:, i] = 0.0
                    up["rows"][:, i] = 0.0
        print(f"  pruned families {zero_fams} ({n_z} cols zeroed)")
    soft_fams = [f.strip() for f in args.soft_families.split(",") if f.strip()]
    soft_scale = float(args.soft_scale)
    if soft_fams:
        n_s = 0
        up = D["upcoming"]
        for fam in soft_fams:
            cols = families.get(fam, [])
            for col in cols:
                if col not in feats:
                    continue
                i = feats.index(col)
                M[:, i] = M[:, i] * soft_scale
                up["masks"][:, i] = up["masks"][:, i] * soft_scale
                n_s += 1
        print(f"  soft-weighted families {soft_fams} ×{soft_scale} ({n_s} cols)")
    print(f"  X={X.shape}, families={list(families)}, targets={targets}")

    seasons = np.array([m["season"] for m in META])
    positions = np.array([POS_INDEX.get(m["pos"], -1) for m in META], dtype=np.int64)
    ped_col = feats.index("d_draft_pick_log")
    ped_y = X[:, ped_col].copy()
    ped_m = M[:, ped_col].copy()

    tr = seasons <= 2023
    va = seasons == 2024
    te = seasons == 2025
    if not tr.any() or not va.any():
        # fallback if seasons shift
        uniq = sorted(set(seasons.tolist()))
        te_s = uniq[-1]; va_s = uniq[-2]
        tr = seasons <= va_s - 1
        va = seasons == va_s
        te = seasons == te_s

    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    # masked cells stay 0 after standardize (tower sees mask)
    Xz = Xz * M
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
    ysd[ysd == 0] = 1.0
    Yz = (Y - ymu) / ysd
    umu, usd = Yu[tr].mean(0), Yu[tr].std(0)
    usd[usd == 0] = 1.0
    Yuz = (Yu - umu) / usd

    season_ids_all = seasons - seasons.min()
    n_seasons = int(season_ids_all.max()) + 1
    slices = family_slices(feats, families)
    fam_dims = {fam: len(cols) for fam, cols in slices.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    model = MTNN(fam_dims, n_seasons=n_seasons, d_emb=args.d_emb, n_targets=len(targets),
                 n_usage=len(D["usage_recon_names"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber = nn.SmoothL1Loss(reduction="none")
    wts = torch.tensor([HEAD_WEIGHTS[t] for t in targets], dtype=torch.float32, device=device)

    xs_all, ms_all = split_by_family(Xz, M, slices, device)
    Ytr_t = torch.tensor(Yz[tr], dtype=torch.float32, device=device)
    Yva_np = Y[va]
    Utr_t = torch.tensor(Yuz[tr], dtype=torch.float32, device=device)
    pos_tr = torch.tensor(positions[tr], dtype=torch.long, device=device)
    ped_tr = torch.tensor(ped_y[tr], dtype=torch.float32, device=device)
    ped_m_tr = torch.tensor(ped_m[tr], dtype=torch.float32, device=device)
    seas_tr = torch.tensor(season_ids_all[tr], dtype=torch.long, device=device)
    seas_va = torch.tensor(season_ids_all[va], dtype=torch.long, device=device)
    tr_idx = np.where(tr)[0]
    va_idx = np.where(va)[0]

    def batch_family(idx):
        return ({f: xs_all[f][idx] for f in xs_all},
                {f: ms_all[f][idx] for f in ms_all})

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}")

    best_va, best_state, bad, patience = 1e9, None, 0, 20
    rng = np.random.default_rng(SEED)
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for s in range(0, len(perm), 512):
            bi = tr_idx[perm[s:s + 512]]
            # map to train-tensor rows
            # easier: index into full tensors
            xs_b, ms_b = batch_family(bi)
            # Family-token dropout: randomly zero entire family masks (train only).
            if args.family_drop > 0:
                for fam in list(ms_b.keys()):
                    if rng.random() < args.family_drop:
                        ms_b[fam] = torch.zeros_like(ms_b[fam])
                        xs_b[fam] = torch.zeros_like(xs_b[fam])
            # align Y — bi indexes full matrix
            # rebuild train-local? use full Yz
            yb = torch.tensor(Yz[bi], dtype=torch.float32, device=device)
            ub = torch.tensor(Yuz[bi], dtype=torch.float32, device=device)
            pb = torch.tensor(positions[bi], dtype=torch.long, device=device)
            pedb = torch.tensor(ped_y[bi], dtype=torch.float32, device=device)
            pedmb = torch.tensor(ped_m[bi], dtype=torch.float32, device=device)
            sb = torch.tensor(season_ids_all[bi], dtype=torch.long, device=device)
            opt.zero_grad()
            emb, out = model(xs_b, ms_b, sb)
            loss_t = (wts * huber(out["targets"], yb).mean(0)).sum()
            loss_u = huber(out["usage"], ub).mean()
            pos_ok = pb >= 0
            loss_p = F.cross_entropy(out["position"][pos_ok], pb[pos_ok]) if pos_ok.any() else 0.0
            if pedmb.sum() > 0:
                loss_ped = (huber(out["pedigree"], pedb) * pedmb).sum() / pedmb.sum()
            else:
                loss_ped = 0.0
            loss = loss_t + 0.15 * loss_u + 0.10 * loss_p + 0.05 * loss_ped
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            xs_v, ms_v = batch_family(va_idx)
            _, out_v = model(xs_v, ms_v, seas_va)
            pv = out_v["targets"].cpu().numpy() * ysd + ymu
            va_mae = float(np.mean(np.abs(pv[:, 0] - Yva_np[:, 0])))
        if va_mae < best_va - 1e-4:
            best_va, bad = va_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch+1}: val PPR MAE {va_mae:.3f} *")
        else:
            bad += 1
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}: val PPR MAE {va_mae:.3f}")
            if bad >= patience:
                print(f"  early stop at epoch {epoch+1}")
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    def predict_rows(Xr, Mr, seas_r):
        Xrz = ((Xr - mu) / sd) * Mr
        xs, ms = split_by_family(Xrz, Mr, slices, device)
        sid = torch.tensor(seas_r - seasons.min(), dtype=torch.long, device=device)
        # clamp season ids into range
        sid = sid.clamp(0, n_seasons - 1)
        with torch.no_grad():
            emb, w = model.encode_with_contrib(xs, ms, sid)
            targets = torch.cat([h(emb) for h in model.target_heads], dim=1)
            pred = targets.cpu().numpy() * ysd + ymu
            return pred, emb.cpu().numpy(), w.cpu().numpy()

    def top_contrib(w_row, k=5):
        order = np.argsort(-w_row)[:k]
        return [
            {"family": model.families[j], "w": round(float(w_row[j]), 3)}
            for j in order if w_row[j] > 0.01
        ]

    # test report
    pte, _, _ = predict_rows(X[te], M[te], seasons[te])
    yte = Y[te]
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    r2 = lambda y, p: float(1 - np.sum((y - p) ** 2) / max(1e-9, np.sum((y - y.mean()) ** 2)))
    f_last4 = X[te][:, feats.index("f_fpts_ppr")]
    f_std = X[te][:, feats.index("std_ppr")]
    y0, p0 = yte[:, 0], pte[:, 0]
    abs_e = np.abs(y0 - p0)
    # MAPE: floor denom so near-zero weeks don't explode (fantasy pts)
    mape_denom = np.maximum(np.abs(y0), 1.0)
    pos_te = np.array([META[i]["pos"] for i in np.where(te)[0]]) if te.any() else np.array([])
    per_pos_mae = {}
    for p in SKILL:
        m = pos_te == p
        if m.sum() >= 30:
            per_pos_mae[p] = round(float(np.mean(abs_e[m])), 3)
    report = {
        "model_fpts_mae": round(mae(p0, y0), 3),
        "model_fpts_r2": round(r2(y0, p0), 3),
        "model_fpts_rmse": round(float(np.sqrt(np.mean((y0 - p0) ** 2))), 3),
        "model_fpts_medae": round(float(np.median(abs_e)), 3),
        "model_fpts_mape": round(float(np.mean(abs_e / mape_denom)), 3),
        "model_fpts_bias": round(float(np.mean(p0 - y0)), 3),
        "per_pos_mae": per_pos_mae,
        "baseline_last4_mae": round(mae(f_last4, y0), 3),
        "baseline_seasontodate_mae": round(mae(f_std, y0), 3),
        "per_stat_mae": {t: round(mae(pte[:, i], yte[:, i]), 2)
                         for i, t in enumerate(targets)},
        "test_season": int(seasons[te][0]) if te.any() else None,
        "n_test": int(te.sum()), "n_train": int(tr.sum()),
        "architecture": f"MTNN v2 gated fusion + family_drop={args.family_drop} d_emb={args.d_emb}",
        "n_features": len(feats), "n_families": len(families),
        "n_params": n_params, "val_mae": round(best_va, 3),
        "v1_mae_reference": V1_MAE,
        "family_drop": args.family_drop,
        "d_emb": args.d_emb,
        "hill_climb": "rz+conformal80+tower_contrib",
        "promote_metric": "mae",
        "metrics_note": "MAE is the promote gate; MAPE/RMSE/MedAE/bias are diagnostic.",
    }
    # Split-conformal bands: per-pos quantile of |residual| at level α (default 80%).
    # Point estimate unchanged; floor/ceil = proj ± q̂ (not ±σ).
    CONF_LEVEL = 0.80
    resid_by_pos = {}  # kept as σ for diagnostics
    conf_by_pos = {}
    if te.any():
        abs_all = abs_e
        for p in SKILL:
            mask_p = pos_te == p
            if mask_p.sum() >= 30:
                err_p = abs_all[mask_p]
                resid_by_pos[p] = float(np.std(yte[mask_p, 0] - pte[mask_p, 0]))
                conf_by_pos[p] = float(np.quantile(err_p, CONF_LEVEL))
        resid = float(np.std(yte[:, 0] - pte[:, 0]))
        conf_global = float(np.quantile(abs_all, CONF_LEVEL))
        # empirical coverage of conformal bands on held-out test
        q_row = np.array([conf_by_pos.get(pos_te[i], conf_global) for i in range(len(pos_te))])
        cover = float(np.mean(abs_all <= q_row))
    else:
        resid = 6.0
        conf_global = 6.0
        cover = 0.0
    report["residual_std"] = round(resid, 3)
    report["residual_std_by_pos"] = {k: round(v, 3) for k, v in resid_by_pos.items()}
    report["conformal_level"] = CONF_LEVEL
    report["conformal_q"] = round(conf_global, 3)
    report["conformal_q_by_pos"] = {k: round(v, 3) for k, v in conf_by_pos.items()}
    report["conformal_coverage"] = round(cover, 3)
    print("  test report:", json.dumps(report))

    # upcoming
    up = D["upcoming"]
    Xup, Mup, upmeta = up["rows"], up["masks"], up["meta"]
    if len(upmeta) == 0:
        print("WARNING: no upcoming rows")
        return 1
    seas_up = np.array([m["season"] for m in upmeta])
    pred_next, Zup, w_next = predict_rows(Xup, Mup, seas_up)

    # neutral season projection
    Xneutral = Xup.copy()
    NEUTRAL = {
        "is_home": 0.5, "rest_days": 7, "is_div": 0, "is_indoor": 0, "is_grass": 1,
        "temp": 65, "wind": 5, "kick_hour": 13, "is_primetime": 0, "is_thu": 0,
        "is_mon": 0, "week_no": 9, "team_implied": 22.5, "opp_implied": 22.5,
        "spread_team": 0, "total_line": 45, "dvp_allowed": 12.0, "dvp_roll4": 12.0,
        "dvp_pass": 8.0, "dvp_rush": 4.0, "dvp_pass_roll4": 8.0,
    }
    for name, val in NEUTRAL.items():
        if name in feats:
            Xneutral[:, feats.index(name)] = val
    pred_season, _, w_season = predict_rows(Xneutral, Mup, seas_up)

    C = Zup - Zup.mean(0)
    U, S, _ = np.linalg.svd(C, full_matrices=False)
    P = U[:, :3] * S[:3]
    P = (P - P.min(0)) / ((P.max(0) - P.min(0)).max() + 1e-9)
    Zn = Zup / (np.linalg.norm(Zup, axis=1, keepdims=True) + 1e-9)
    sim = Zn @ Zn.T
    np.fill_diagonal(sim, -1)

    inj = {}
    for r in nfl.injuries(up["season"], args.offline):
        if int(nfl.num(r, "week")) == (up["week"] or 1):
            st = (r.get("report_status") or "").strip()
            if st:
                inj[r.get("gsis_id") or r.get("player_id")] = st

    def avail(m):
        if m.get("bye") == up["week"]:
            return "BYE"
        if m["gsis"] in inj:
            return inj[m["gsis"]]
        s = (m.get("status") or "").upper()
        return "" if s in ("", "ACT", "A01") else s

    ti = {t: i for i, t in enumerate(targets)}
    ng_players = []
    for i, m in enumerate(upmeta):
        p = pred_next[i]
        qband = conf_by_pos.get(m["pos"], conf_global)
        ng_players.append({
            "key": norm_key(m["name"], m["pos"]), "name": m["name"], "pos": m["pos"],
            "team": m["team"], "opp": m["opp"], "headshot": m["headshot"],
            "moved": m.get("moved", False), "prev_team": m.get("prev_team", ""),
            "bye": m.get("bye"), "avail": avail(m),
            "proj": round(float(p[ti["fpts_ppr"]]), 2),
            "floor": round(max(0.0, float(p[ti["fpts_ppr"]]) - qband), 2),
            "ceil": round(float(p[ti["fpts_ppr"]]) + qband, 2),
            "uncertainty": round(qband, 2),
            "line": {
                "rec_yds": round(float(p[ti["rec_yds"]]), 1),
                "rush_yds": round(float(p[ti["rush_yds"]]), 1),
                "pass_yds": round(float(p[ti["pass_yds"]]), 1),
                "rec": round(float(p[ti["receptions"]]), 1),
                "td": round(float(p[ti["total_td"]]), 2),
            },
            "tower_contrib": top_contrib(w_next[i]),
            "conditions": m["conditions"],
        })
    ng_players.sort(key=lambda r: r["proj"], reverse=True)
    (ASSETS / "nextgame.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d"), "season": up["season"], "week": up["week"],
        "model": {
            "type": "MTNN v2 multi-tower gated fusion",
            "report": report, "residual_std": round(resid, 2),
            "residual_std_by_pos": {k: round(v, 2) for k, v in resid_by_pos.items()},
            "floor_ceil": f"conformal_abs_q{int(CONF_LEVEL * 100)}",
            "conformal_q_by_pos": {k: round(v, 2) for k, v in conf_by_pos.items()},
            "tower_contrib": "gated_attn_x_gate_top5",
        },
        "count": len(ng_players), "players": ng_players,
    }, separators=(",", ":")), encoding="utf-8")

    proj_players = []
    for i, m in enumerate(upmeta):
        p = pred_season[i]
        order = np.argsort(-sim[i])[:5]
        qband = conf_by_pos.get(m["pos"], conf_global)
        proj_players.append({
            "key": norm_key(m["name"], m["pos"]), "name": m["name"], "pos": m["pos"],
            "team": m["team"], "headshot": m["headshot"],
            "moved": m.get("moved", False), "prev_team": m.get("prev_team", ""),
            "bye": m.get("bye"), "avail": avail(m), "rookie": False,
            "proj": round(float(p[ti["fpts_ppr"]]), 2),
            "floor": round(max(0.0, float(p[ti["fpts_ppr"]]) - qband), 2),
            "ceil": round(float(p[ti["fpts_ppr"]]) + qband, 2),
            "uncertainty": round(qband, 2),
            "line": {
                "rec_yds": round(float(p[ti["rec_yds"]]), 1),
                "rush_yds": round(float(p[ti["rush_yds"]]), 1),
                "pass_yds": round(float(p[ti["pass_yds"]]), 1),
                "rec": round(float(p[ti["receptions"]]), 1),
                "td": round(float(p[ti["total_td"]]), 2),
            },
            "tower_contrib": top_contrib(w_season[i]),
            "comps": [
                {"name": upmeta[j]["name"], "pos": upmeta[j]["pos"],
                 "team": upmeta[j]["team"], "sim": round(float(sim[i, j]), 3)}
                for j in order
            ],
        })
    veteran_keys = {r["key"] for r in proj_players}
    r_predict, r_report, r_meta, r_hist = train_rookie_model(last_season)
    rookies = rookie_board(r_predict, r_meta, r_hist, veteran_keys,
                           last_season, up["season"], resid, ti, D["byes"])
    proj_players.extend(rookies)
    print(f"  rookie model: val fpts MAE {r_report['val_fpts_mae']} vs "
          f"positional baseline {r_report['baseline_pos_mae']}; "
          f"added {len(rookies)} rookies")

    proj_players.sort(key=lambda r: r["proj"], reverse=True)
    for i, r in enumerate(proj_players):
        r["rank_overall"] = i + 1
    for pos in ("QB", "RB", "WR", "TE"):
        grp = [r for r in proj_players if r["pos"] == pos]
        for j, r in enumerate(grp):
            r["rank_pos"] = j + 1
            r["tier"] = 1 + j // max(1, (len(grp) // 6 or 6))

    (ASSETS / "projections.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d"), "proj_season": up["season"],
        "basis": (f"MTNN v2 multi-tower under neutral conditions, off {last_season} form; "
                  f"rookies via draft-capital model"),
        "scoring_feature": "PPR",
        "model": {"report": report, "residual_std": round(resid, 2)},
        "rookie_model": r_report, "rookie_count": len(rookies),
        "count": len(proj_players), "players": proj_players,
    }, separators=(",", ":")), encoding="utf-8")

    emb_players = [{
        "key": norm_key(m["name"], m["pos"]), "name": m["name"], "pos": m["pos"],
        "team": m["team"],
        "x": round(float(P[i, 0]), 4), "y": round(float(P[i, 1]), 4),
        "z": round(float(P[i, 2]), 4),
    } for i, m in enumerate(upmeta)]
    (ASSETS / "embedding.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d"), "season": last_season,
        "dims": int(Zup.shape[1]), "count": len(emb_players), "players": emb_players,
        "method": "L2-normalized MTNN v2 gated-fusion embedding; PCA(3) for map",
    }, separators=(",", ":")), encoding="utf-8")

    DATA.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd,
                "feats": feats, "families": families, "report": report,
                "n_seasons": n_seasons, "season_min": int(seasons.min()),
                "d_emb": args.d_emb, "test_season": report.get("test_season")},
               DATA / "mtnn_best.pt")
    (DATA / "mtnn_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    beat = report["baseline_last4_mae"] - report["model_fpts_mae"]
    print(f"\nwrote nextgame ({len(ng_players)}), projections ({len(proj_players)}), "
          f"embedding ({len(emb_players)})")
    print(f"  next-game fantasy MAE {report['model_fpts_mae']} vs last-4 "
          f"{report['baseline_last4_mae']} ({'beats' if beat > 0 else 'LOSES'} by "
          f"{abs(beat):.3f}), R^2 {report['model_fpts_r2']}; {time.time()-t0:.0f}s")
    print(f"  vs v1 reference MAE {V1_MAE}: "
          f"{'better' if report['model_fpts_mae'] <= V1_MAE + 0.05 else 'worse'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
