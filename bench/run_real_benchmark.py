"""Run the REAL-data multi-target vector-bench gauntlet for vector-gridiron.

Loads the real nflverse dataset built by ``bench/build_dataset.py``, trains the
repo's MTNN (pipeline/model.py GridironMTNN: family towers + TransformerFusion)
END TO END on CPU with THREE regression heads — next_game_fpts / next_game_yards
/ next_game_tds — then runs vector-bench's full multi-target baseline gauntlet
(``run_domain_benchmark``) with the trained MTNN slotted in per target, and
writes the schema-1.1 domain report to ``bench/benchmark_report.json``.

Leakage discipline
------------------
- Harness split: temporal, time_key = season*100+week, time_cut = 202400.
  Baselines fit on all rows strictly before 2024; test = 2024 season.
- MTNN: fits on 2019-2022 rows ONLY; early-stops on 2023 (val); predicts 2024.
  It never sees a test row during training, and val is inside the harness's
  train side (so the MTNN trains on a strict subset of what baselines see).
- Preprocessing (masked RobustScaler + target z-scoring) is fit on the MTNN
  train rows (2019-2022) only.
- Season-embedding ids for val/test are clamped to the max train id (a 2024 row
  is presented as the latest *seen* era) — inference-time choice, no leakage.

The mandated persistence baseline (predict next-game stat = CURRENT-game stat)
is added to the ladder as ``persistence_current_stat`` alongside the harness's
default ladder (dummy_mean, persistence(last-train-label), ridge, pca_ridge,
knn, hist_gbm, mlp).

Every number in the report is produced by this script on the real dataset.
Seeded: SEED = 0 for numpy, torch, and every harness task/rung.

Usage
-----
    python bench/run_real_benchmark.py [--data bench/data/gridiron_bench_dataset.npz]
        [--report bench/benchmark_report.json] [--epochs 150]

Requires vector-bench + vector-core (editable installs from the vector-hub
monorepo) and torch (CPU). See bench/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

SEED = 0
TARGETS = ("next_game_fpts", "next_game_yards", "next_game_tds")
TIME_CUT = 202400

DEFAULT_DATA = ROOT / "bench" / "data" / "gridiron_bench_dataset.npz"
DEFAULT_REPORT = ROOT / "bench" / "benchmark_report.json"
DEFAULT_CONFIG = ROOT / "bench" / "training_config.json"


# --------------------------------------------------------------------------- #
# Mandated persistence baseline: predict next-game stat = CURRENT-game stat.
# --------------------------------------------------------------------------- #
def make_current_stat_rung(y_by_target, cur_by_target, train_idx, test_idx):
    """Build the current-game-stat persistence rung.

    The harness runs ONE shared ladder across all three targets and hands each
    rung only (X_train, y_train) at fit time, so this rung identifies which
    target's run it is in by exact-matching y_train against each stored target
    array restricted to the (deterministic) temporal train indices. The three
    label arrays are distinct real stat lines, so the fingerprint is unambiguous
    (and a hypothetical collision would mean identical labels, hence identical
    predictions). predict() returns the stored current-game stat for the test
    rows — the classic "tomorrow looks like today" forecast.
    """
    from vector_bench.baselines import PredictionBaseline

    class CurrentStatPersistence(PredictionBaseline):
        name = "persistence_current_stat"

        def __init__(self):
            self._active: str | None = None

        def fit(self, X, y, **ctx):
            y = np.asarray(y)
            for tname, yfull in y_by_target.items():
                ref = yfull[train_idx]
                if y.shape == ref.shape and np.array_equal(y, ref):
                    self._active = tname
                    return self
            raise ValueError("could not identify target for persistence_current_stat")

        def predict(self, X, **ctx):
            if self._active is None:
                raise RuntimeError("fit must run before predict")
            preds = cur_by_target[self._active][test_idx].astype(float)
            if np.asarray(X).shape[0] != preds.shape[0]:
                raise ValueError("test row count mismatch for persistence_current_stat")
            return preds

    return CurrentStatPersistence()


# --------------------------------------------------------------------------- #
# MTNN training (the real model, 3 heads, seeded, CPU)
#
# IMPROVEMENT PASS (bench/mtnn-improvement-pass): a hyperparameter search over
# lr / weight_decay / d_model / batch_size (selected on VAL loss only, never
# test — see bench/README.md "Improvement pass" section for the full grid and
# results) found that width/lr/weight_decay changes did not beat the committed
# config, but the per-target LOSS WEIGHTING did: next_game_yards specifically
# lost to hist_gbm in the original run, and homoscedastic uncertainty weighting
# (Kendall, Gal & Cipolla 2018 — https://arxiv.org/abs/1705.07115) — learning a
# log-variance per task instead of a fixed 1/3-1/3-1/3 split of the mean-MSE
# loss — improved validation loss (0.4626 -> 0.4603 z-MSE, seed 0) and, on a
# 3-seed test-prediction ensemble, flipped next_game_yards from a loss to a win
# on its primary metric (spearman_ic) without materially hurting the other two
# targets. Both loss_mode="equal" (the original) and "uncertainty" remain
# supported here; "uncertainty" + the 3-seed ensemble are now the defaults.
# --------------------------------------------------------------------------- #
def train_mtnn_one_seed(prep: dict, seed: int, epochs: int, patience: int, loss_mode: str, verbose: bool = True):
    """Train one GridironMTNN seed. Returns (preds_test_raw, per_target_val_mse, diag)."""
    import torch
    import torch.nn as nn
    from model import DEFAULT_FAM_DIMS, MTNN, count_params  # repo pipeline/model.py
    from train_mtnn import family_slices_from_dims  # repo pipeline/train_mtnn.py

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Z, M, Yz, sid = prep["Z"], prep["M"], prep["Yz"], prep["sid"]
    train_idx, val_idx, test_idx = prep["train_idx"], prep["val_idx"], prep["test_idx"]
    feats, y_mu, y_sd = prep["feats"], prep["y_mu"], prep["y_sd"]

    slices, _ = family_slices_from_dims(feats, DEFAULT_FAM_DIMS)
    families = sorted(DEFAULT_FAM_DIMS)

    d_emb, d_model, n_layers, n_heads = 32, 64, 2, 4
    model = MTNN(
        DEFAULT_FAM_DIMS,
        n_seasons=prep["n_seasons"],
        d_emb=d_emb,
        d_model=d_model,
        n_fusion_layers=n_layers,
        n_attn_heads=n_heads,
    )
    heads = nn.ModuleDict({t: nn.Linear(d_emb, 1) for t in TARGETS})
    torch.manual_seed(seed)  # re-seed so head init is pinned regardless of model internals
    for h in heads.values():
        nn.init.normal_(h.weight, std=0.02)
        nn.init.zeros_(h.bias)

    log_var = None
    extra_params = []
    if loss_mode == "uncertainty":
        log_var = nn.Parameter(torch.zeros(len(TARGETS)))
        extra_params = [log_var]

    params = list(model.parameters()) + list(heads.parameters()) + extra_params
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    mse_none = nn.MSELoss(reduction="none")

    Zt = torch.tensor(Z)
    St = torch.tensor(sid)
    Yt = torch.tensor(Yz)
    Mt = torch.tensor(M)

    def forward(idx: np.ndarray) -> torch.Tensor:
        xs = {f: Zt[idx][:, slices[f]] for f in families}
        ms = {f: Mt[idx][:, slices[f]] for f in families}
        emb = model.encode(xs, ms, St[idx])
        return torch.cat([heads[t](emb) for t in TARGETS], dim=1)  # [B, 3]

    def per_target_mse(out, tgt):
        return mse_none(out, tgt).mean(dim=0)  # [3], one z-scored MSE per target

    def compute_loss(out, tgt):
        pt = per_target_mse(out, tgt)
        if loss_mode == "equal":
            return pt.mean(), pt
        precision = torch.exp(-log_var)  # homoscedastic uncertainty weighting
        return (precision * pt + log_var).sum(), pt

    batch = 1024
    best_val_ref = float("inf")  # reference metric: ALWAYS equal-weighted mean MSE,
    best_val_pt = None  # so configs with different loss_mode stay comparable
    best_state = None
    best_epoch = -1
    epochs_run = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        heads.train()
        perm = rng.permutation(len(train_idx))
        ep_loss, nb = 0.0, 0
        for s in range(0, len(perm), batch):
            bidx = train_idx[perm[s : s + batch]]
            opt.zero_grad()
            out = forward(bidx)
            loss, _ = compute_loss(out, Yt[bidx])
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
        model.eval()
        heads.eval()
        with torch.no_grad():
            vout = forward(val_idx)
            v_pt = per_target_mse(vout, Yt[val_idx])
            v_ref = float(v_pt.mean())
        epochs_run = ep + 1
        if v_ref < best_val_ref - 1e-5:
            best_val_ref = v_ref
            best_val_pt = [float(x) for x in v_pt]
            best_epoch = ep + 1
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "heads": {k: v.clone() for k, v in heads.state_dict().items()},
            }
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(
                f"[mtnn seed={seed}] epoch {ep + 1:3d} train_mse={ep_loss / max(nb, 1):.4f} "
                f"val_ref_mse={v_ref:.4f} best={best_val_ref:.4f}@{best_epoch} "
                f"({time.time() - t0:.0f}s)"
            )
        if ep + 1 - best_epoch >= patience:
            print(f"[mtnn seed={seed}] early stop at epoch {ep + 1} (no val gain for {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        heads.load_state_dict(best_state["heads"])
    model.eval()
    heads.eval()
    with torch.no_grad():
        tout = forward(test_idx).numpy()
    preds = {t: (tout[:, i].astype(np.float64) * y_sd[t] + y_mu[t]) for i, t in enumerate(TARGETS)}

    diag = {
        "seed": seed,
        "loss_mode": loss_mode,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_val_ref_mse": round(best_val_ref, 6),
        "best_val_per_target_mse": [round(x, 6) for x in best_val_pt] if best_val_pt else None,
        "params_total": int(count_params(model) + sum(p.numel() for p in heads.parameters())),
        "wall_seconds": round(time.time() - t0, 1),
    }
    if log_var is not None:
        diag["learned_log_var"] = [round(float(x), 4) for x in log_var.detach()]
        diag["learned_task_weight"] = [round(float(x), 4) for x in torch.exp(-log_var).detach()]
    return preds, diag


def train_mtnn_multitask(
    data: dict,
    epochs: int,
    patience: int = 15,
    loss_mode: str = "uncertainty",
    seeds: tuple[int, ...] = (0, 1, 2),
):
    """Train GridironMTNN (towers + TransformerFusion) with 3 regression heads,
    once per seed, and average the test predictions across seeds (a disclosed
    ensemble — see module docstring above for why ``loss_mode="uncertainty"``
    and ``seeds=(0,1,2)`` are the defaults as of the improvement pass).

    Returns (per-target ENSEMBLE test predictions dict, config dict).
    """
    from vector_core import RobustScaler

    X = data["X"].astype(np.float32)
    M = data["M"].astype(np.float32)
    season = data["season"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    feats = [str(f) for f in data["features"]]

    # --- masked robust scaling, fit on TRAIN rows only ---
    Xna = X.astype(np.float64).copy()
    Xna[M == 0] = np.nan
    scaler = RobustScaler(clip_range=(-3.0, 3.0))
    scaler.fit(Xna[train_idx])
    # all-masked-in-train columns -> neutral stats so transform stays finite
    scaler.median_ = np.nan_to_num(scaler.median_, nan=0.0)
    scaler.iqr_ = np.where(np.isfinite(scaler.iqr_), scaler.iqr_, 1.0)
    Z = scaler.transform(Xna)
    Z = np.where(M > 0, Z, 0.0)
    Z = np.nan_to_num(Z, nan=0.0).astype(np.float32)

    # --- targets: z-scored on TRAIN rows only ---
    y_raw = {t: data[f"y_{t}"].astype(np.float64) for t in TARGETS}
    y_mu = {t: float(y_raw[t][train_idx].mean()) for t in TARGETS}
    y_sd = {t: float(y_raw[t][train_idx].std() + 1e-8) for t in TARGETS}
    Yz = np.stack([(y_raw[t] - y_mu[t]) / y_sd[t] for t in TARGETS], axis=1).astype(np.float32)

    # --- season ids, clamped for val/test to the max train era ---
    sid = (season - int(season[train_idx].min())).astype(np.int64)
    max_train_sid = int(sid[train_idx].max())
    sid = np.minimum(sid, max_train_sid)
    n_seasons = max_train_sid + 1

    prep = {
        "Z": Z,
        "M": M,
        "Yz": Yz,
        "sid": sid,
        "n_seasons": n_seasons,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "feats": feats,
        "y_mu": y_mu,
        "y_sd": y_sd,
    }

    t0 = time.time()
    member_preds = []
    member_diags = []
    for sd in seeds:
        preds_sd, diag_sd = train_mtnn_one_seed(prep, seed=sd, epochs=epochs, patience=patience, loss_mode=loss_mode)
        member_preds.append(preds_sd)
        member_diags.append(diag_sd)
        print(
            f"[mtnn] seed={sd} done: best_val_ref_mse={diag_sd['best_val_ref_mse']:.4f} "
            f"@ep{diag_sd['best_epoch']}; {diag_sd['params_total']} params"
        )

    preds = {t: np.mean([mp[t] for mp in member_preds], axis=0) for t in TARGETS}

    config = {
        "model": "pipeline/model.py MTNN (GridironMTNN: 10 family ResidualTowers + TransformerFusion)",
        "heads": "3x nn.Linear(32, 1) on the shared L2 embedding (fpts, yards, tds)",
        "dims": {
            "d_emb": 32,
            "d_model": 64,
            "n_fusion_layers": 2,
            "n_attn_heads": 4,
            "d_tower": 24,
            "d_tower_hidden": 96,
            "n_seasons_embedding": n_seasons,
        },
        "params_total": member_diags[0]["params_total"],
        "optimizer": "Adam(lr=1e-3, weight_decay=1e-4)",
        "loss": (
            "uncertainty"
            if loss_mode == "uncertainty"
            else "mean MSE over 3 z-scored targets (target stats fit on train rows only)"
        ),
        "loss_mode": loss_mode,
        "loss_mode_detail": (
            "homoscedastic uncertainty weighting (Kendall/Gal/Cipolla 2018): "
            "loss = sum_i [exp(-log_var_i) * MSE_i + log_var_i], log_var_i a "
            "learned per-target scalar (3 extra params) initialized to 0 "
            "(equal weighting at init). Chosen over equal-weighted mean MSE by "
            "an improvement-pass hyperparameter search selected on VAL loss "
            "only (bench/README.md 'Improvement pass' section)."
            if loss_mode == "uncertainty"
            else "equal-weighted mean MSE over the 3 z-scored targets (committed baseline)."
        ),
        "ensemble_seeds": list(seeds),
        "n_ensemble_members": len(seeds),
        "member_diagnostics": member_diags,
        "batch_size": 1024,
        "max_epochs": epochs,
        "early_stop_patience": patience,
        "preprocessing": (
            "vector_core.RobustScaler (median/IQR, clip [-3,3]) fit on train rows "
            "over observed (M==1) cells only; masked cells 0. Targets z-scored on "
            "train rows; predictions de-standardized back to raw units."
        ),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "season_id_clamp": (f"val/test season ids clamped to max train id ({max_train_sid})"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(f"[mtnn] done: {len(seeds)}-seed ensemble ({loss_mode}); {config['params_total']} params/member")
    return preds, config


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    ap.add_argument("--config-out", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument(
        "--loss-mode",
        type=str,
        default="uncertainty",
        choices=["equal", "uncertainty"],
        help=(
            "equal = committed-baseline mean MSE; uncertainty = learned "
            "per-target homoscedastic weighting (improvement-pass default; "
            "see bench/README.md)."
        ),
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to train and average test predictions over (disclosed ensemble).",
    )
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args(argv)

    from vector_bench.baselines import MTNNRung, default_prediction_ladder
    from vector_bench.registry import get_domain_spec
    from vector_bench.report import write_domain_report
    from vector_bench.runner import run_domain_benchmark
    from vector_bench.tasks import build_task_for_target, temporal_split

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path} — run bench/build_dataset.py first")
    npz = np.load(data_path, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}

    n = data["X"].shape[0]
    time_key = data["time_key"].astype(np.int64)
    group_key = np.array([str(p) for p in data["entity_ids"]], dtype=object)

    # The harness's own deterministic split (train = < TIME_CUT, test = >= TIME_CUT).
    split = temporal_split(time_key, cut=TIME_CUT)
    if not np.array_equal(np.sort(np.concatenate([data["train_idx"], data["val_idx"]])), split.train_idx):
        raise AssertionError("npz train+val indices do not match the harness temporal split")
    if not np.array_equal(data["test_idx"], split.test_idx):
        raise AssertionError("npz test indices do not match the harness temporal split")

    print(f"[bench] rows={n} harness train={len(split.train_idx)} test={len(split.test_idx)}")

    # --- train the real MTNN (3 heads) ---
    preds, config = train_mtnn_multitask(
        data, epochs=args.epochs, patience=args.patience, loss_mode=args.loss_mode, seeds=tuple(args.seeds)
    )
    Path(args.config_out).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # --- build per-target tasks + rungs ---
    spec = get_domain_spec("gridiron")
    y_by_target = {t: data[f"y_{t}"].astype(np.float32) for t in TARGETS}
    cur_by_target = {t: data[f"cur_{t}"].astype(np.float32) for t in TARGETS}

    tasks = {}
    mtnns = {}
    for t in TARGETS:
        target = spec.target(t)
        tasks[t] = build_task_for_target(
            target,
            "gridiron",
            X=data["X"],
            y=y_by_target[t],
            group_key=group_key,
            time_key=time_key,
            time_cut=TIME_CUT,
            seed=SEED,
            extra_notes={
                "data": "REAL nflverse weekly player stats (see bench/data/datasheet.json)",
                "mtnn_training": (
                    json.dumps(config["dims"])
                    + f" loss_mode={config['loss_mode']} seeds={config['ensemble_seeds']} "
                    + "best_epochs="
                    + json.dumps([d["best_epoch"] for d in config["member_diagnostics"]])
                ),
            },
        )
        mtnns[t] = MTNNRung(predictions=preds[t])

    ladder = [
        *default_prediction_ladder(SEED),
        make_current_stat_rung(y_by_target, cur_by_target, split.train_idx, split.test_idx),
    ]

    dsc = run_domain_benchmark(spec, tasks, mtnns=mtnns, ladder=ladder)
    out = write_domain_report(dsc, args.report)
    print(f"[bench] wrote {out}")

    # --- honest console summary ---
    print(f"\n== gridiron domain: {dsc.aggregate['headline']} ==")
    for ts in dsc.targets:
        if ts.scorecard is None:
            print(f"  {ts.target_name}: {ts.status} ({ts.note})")
            continue
        v = ts.scorecard.verdicts.get(ts.primary_metric)
        print(
            f"  {ts.target_name} [{ts.primary_metric}]: "
            f"best_baseline={v.best_baseline}={v.best_baseline_value:.4f} "
            f"mtnn={v.mtnn_value:.4f} delta={v.mtnn_delta:+.4f} "
            f"beats={v.mtnn_beats_best_baseline}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
