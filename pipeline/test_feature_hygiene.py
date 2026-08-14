"""Feature-hygiene gates -- same discipline as vector-hoops'
pipeline/test_feature_hygiene.py, adapted for this repo's schema (weekly
rows, targets live in a separate Y array rather than mixed into Z as an
input family). Turns audit_features.py's one-time findings into a pinned
regression gate: a NEW duplicate, leak, or dead column fails the run; the
already-investigated ones stay allowlisted with a reason.

Run:  python pipeline/test_feature_hygiene.py    (exit 0 = all gates pass)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pipeline" / "data"
MATRIX = DATA / "train_matrix.npz"
MANIFEST = DATA / "feature_manifest.json"

DUP_R = 0.98
LEAK_R = 0.90
MIN_OVERLAP = 400
MIN_COVERAGE = 0.01
NEAR_CONST_STD = 0.01

# Redundant input pairs found by audit_features.py (2026-07-30) and judged
# harmless -- add here only with a reason, not to silence a real duplicate.
KNOWN_DUPLICATES = {
    # Carries/targets counted twice: once as a trailing form feature, once as
    # the opportunity-share input it's derived from. Same shape as hoops'
    # OREB~OREB_PCT -- both kept because they feed different towers.
    "f_carries~o_rush_attempt",
    "f_targets~o_rec_attempt",
    # QB time-to-throw is close to a QB indicator by construction (non-QBs
    # get a fixed fallback value) -- structural, not duplicated sourcing.
    "is_QB~n_ngs_ttt",
    # Trailing pass yards and pass attempts move together for any QB with a
    # stable per-attempt average -- correlated by football, not by pipeline bug.
    "f_pass_yds~f_pass_att",
}

# Input-vs-target correlations found by audit_features.py and judged expected:
# trailing form features are literally lagged copies of the stat they're
# trying to predict, so a high correlation with the real target is the
# feature working as designed, not a leak into the label itself.
KNOWN_LEAKS = {
    "pass_yds<-n_ngs_ttt",
    "pass_yds<-f_pass_yds",
    "pass_yds<-f_pass_att",
}

# Rare-event binary flags (player officially out/doubtful, 4+ games missed)
# that are near-constant simply because the event is rare in rows where the
# player actually played that week. Measured non-degenerate (values in
# {0,1}, coverage 89%, occurs a nonzero number of times) -- not dead.
KNOWN_NEAR_CONSTANT = {
    "a_inj_out",
    "a_inj_doubtful",
    "a_games_missed_4",
}

FAILURES: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def masked_corr(a, b, ma, mb) -> tuple[float, int]:
    both = (ma > 0) & (mb > 0)
    n = int(both.sum())
    if n < MIN_OVERLAP:
        return 0.0, n
    x, y = a[both].astype(np.float64), b[both].astype(np.float64)
    sx, sy = x.std(), y.std()
    if sx < 1e-9 or sy < 1e-9:
        return 0.0, n
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)), n


def main() -> None:
    if not MATRIX.exists() or not MANIFEST.exists():
        print("train_matrix.npz / feature_manifest.json missing -- run build_features.py")
        sys.exit(1)

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    m = np.load(MATRIX, allow_pickle=True)
    Z, M = m["Z"], m["mask"]
    Y = m["Y"] if "Y" in m else None
    targets: list[str] = man.get("targets", [])
    feats: list[str] = man["features"]
    fam_of: dict[str, str] = man["families"]

    print("shape")
    check(Z.shape[1] == len(feats), f"matrix width == manifest features ({len(feats)})")
    check(Z.shape == M.shape, "values and mask same shape")

    print("no dead columns")
    dead = []
    for j, f in enumerate(feats):
        if f in KNOWN_NEAR_CONSTANT:
            continue
        obs = M[:, j] > 0
        if obs.mean() < MIN_COVERAGE:
            dead.append(f"{f} (coverage {obs.mean():.4f})")
        elif obs.sum() >= MIN_OVERLAP and float(Z[obs, j].std()) < NEAR_CONST_STD:
            dead.append(f"{f} (near-constant)")
    check(
        not dead,
        f"every feature carries signal{'' if not dead else ': ' + ', '.join(dead[:5])}",
    )

    print("no new duplicate input pairs")
    dups = []
    for j in range(len(feats)):
        for k in range(j + 1, len(feats)):
            r, _ = masked_corr(Z[:, j], Z[:, k], M[:, j], M[:, k])
            if abs(r) >= DUP_R:
                dups.append(f"{feats[j]}~{feats[k]} r={r:+.4f}")
    unknown_dups = [d for d in dups if d.split(" r=")[0] not in KNOWN_DUPLICATES]
    check(
        not unknown_dups,
        f"no new duplicate pairs |r|>={DUP_R}"
        f"{'' if not unknown_dups else ': ' + ', '.join(unknown_dups[:5])}",
    )

    print("no new input leaks a real target")
    leaks = []
    if Y is not None and targets:
        tmask_all = np.ones(Y.shape[0], dtype=np.float32)
        for ti, tname in enumerate(targets):
            tcol = Y[:, ti]
            for k, f in enumerate(feats):
                r, _ = masked_corr(tcol, Z[:, k], tmask_all, M[:, k])
                if abs(r) >= LEAK_R:
                    leaks.append(f"{tname}<-{f} r={r:+.4f}")
    unknown_leaks = [d for d in leaks if d.split(" r=")[0] not in KNOWN_LEAKS]
    check(
        not unknown_leaks,
        f"no new input within |r|>={LEAK_R} of a real target"
        f"{'' if not unknown_leaks else ': ' + ', '.join(unknown_leaks[:5])}",
    )

    print("families intact")
    fam_cols: dict[str, list[int]] = defaultdict(list)
    for j, f in enumerate(feats):
        fam_cols[fam_of.get(f, "?")].append(j)
    check("?" not in fam_cols, "every feature has a family")
    check(len(fam_cols) >= 10, f"family count sane ({len(fam_cols)})")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} feature-hygiene gate(s) FAILED")
        sys.exit(1)
    print("all feature-hygiene gates passed")


if __name__ == "__main__":
    main()
