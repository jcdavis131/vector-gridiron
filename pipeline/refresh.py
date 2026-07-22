"""Weekly auto-refresh: pull the latest nflverse data, rebuild every artifact,
and redeploy to Vercel. Safe to run unattended (this is what the scheduled task
calls). Season rollover is automatic — build_vectors ranges to the current year
and train_models projects (latest published season + 1), so once nflverse posts
stats_player_week_<new year>.csv it flows through with no code change.

  python pipeline/refresh.py

Wire it weekly (Windows) with:
  schtasks /Create /TN VectorGridironWeeklyRefresh /SC WEEKLY /D TUE /ST 10:00 ^
    /TR "cmd /c cd /d C:\\Users\\jcdav\\vector-gridiron && python pipeline\\refresh.py"
(Tuesday: nflverse finalizes the prior NFL week Mon night / Tue.)
"""

from __future__ import annotations

import subprocess
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache"
LOG = ROOT / "pipeline" / "refresh.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def invalidate() -> None:
    """Drop the volatile feeds so fresh weekly data is refetched; older complete
    seasons stay cached (they never change)."""
    yr = date.today().year
    names = ["games.csv", "players.csv"]
    for y in (yr, yr - 1):
        names += [
            f"stats_player_week_{y}.csv", f"snap_counts_{y}.csv",
            f"depth_charts_{y}.csv", f"injuries_{y}.csv",
            f"ep_weekly_{y}.csv",
            f"ngs_{y}_receiving.csv.gz", f"ngs_{y}_rushing.csv.gz",
            f"ngs_{y}_passing.csv.gz",
            f"advstats_week_rec_{y}.csv", f"advstats_week_rush_{y}.csv",
            f"advstats_week_pass_{y}.csv",
        ]
    for name in names:
        p = CACHE / name
        if p.exists():
            try:
                p.unlink(); log(f"invalidated {name}")
            except OSError as e:
                log(f"could not remove {name}: {e}")


def deploy() -> None:
    log("deploying to Vercel ...")
    try:
        r = subprocess.run("vercel deploy --prod --yes", cwd=str(ROOT), shell=True,
                           capture_output=True, text=True, timeout=900)
        lines = (r.stdout + "\n" + r.stderr).splitlines()
        prod = [l.strip() for l in lines if "Production:" in l or ".vercel.app" in l]
        log("deploy: " + (prod[-1] if prod else f"exit {r.returncode}"))
    except Exception as e:
        log(f"deploy failed ({e}); run `vercel deploy --prod --yes` in {ROOT} manually")


def run() -> None:
    t0 = time.time()
    log("=== weekly refresh start ===")
    invalidate()
    import build_vectors
    import train_mtnn
    import build_adp
    log("building vectors / profiles / grades ...")
    build_vectors.main()
    log("training MTNN v2 (multi-tower) + rookie model ...")
    train_mtnn.main()
    log("walk-forward rank backtest ...")
    try:
        import build_backtest
        build_backtest.main()
    except Exception as e:
        log(f"backtest failed (non-fatal, prior artifact kept): {e}")
    log("pulling consensus ADP ...")
    try:
        build_adp.main()
    except Exception as e:
        log(f"ADP refresh failed (non-fatal): {e}")
    log("building K/DST projections ...")
    try:
        import build_kdst
        build_kdst.build()
    except Exception as e:
        log(f"K/DST build failed (non-fatal): {e}")
    log("seeding lookback league (grades + narratives) ...")
    try:
        import build_lookback
        build_lookback.main()
    except Exception as e:
        log(f"lookback build failed (non-fatal): {e}")
    deploy()
    log(f"=== weekly refresh done in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    run()
