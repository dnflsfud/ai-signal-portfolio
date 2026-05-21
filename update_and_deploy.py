"""End-to-end pipeline: refresh data → backtest → build dashboard payload → push to cc2-dashboard repo.

새 데이터(`data/ai_signal_data.xlsx`)가 들어오면 이 스크립트 한 번만 돌리면
Streamlit Cloud까지 자동 배포된다.

흐름:
    1. data 파일 sanity check (파일 존재 + 마지막 영업일 freshness 검증)
    2. backtest 실행
        - --mode full (default) → run_variant.py (production variant 풀 백테스트)
                                  결과를 outputs/<variant_label>/ 에 저장 후
                                  outputs/baseline_v4/ 로 자동 promote
        - --mode incremental    → daily_update.py (빠름; baseline pkl은 안 갱신함에 주의)
    3. scripts/build_dashboard_data.py 로 dashboard_data.pkl(~3MB) 생성
    4. (옵션) 로컬 streamlit 스모크 실행
    5. cc2-dashboard repo로 파일 동기화 → git add/commit/push
       → Streamlit Cloud가 push를 감지해서 30초 안에 재배포

사용 예:
    # 디폴트 — production variant 풀 백테스트 + 자동 배포 (~3-4분)
    python update_and_deploy.py

    # 빠른 증분 모드 — daily 신호만 갱신 (대시보드 backtest pkl은 갱신 안 됨)
    python update_and_deploy.py --mode incremental

    # 풀 백테스트지만 다른 variant
    python update_and_deploy.py --variant variants/iter15_65tkr_reb21.yaml

    # 로컬에서 결과만 확인하고 push는 보류
    python update_and_deploy.py --no-push

    # 로컬 streamlit으로 스모크 확인까지
    python update_and_deploy.py --smoke

환경 변수(선택):
    CC2_DASHBOARD_REPO   배포 대상 dashboard repo 경로
                         (default: C:/Users/westl/Documents/cc2-dashboard)
    CC2_RUN_DIR          backtest 결과 디렉터리 (canonical promote 대상)
                         (default: outputs/baseline_v4)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_FILE = Path(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx")
# Production variant. 2026-05-19 v2 cutover (final-v1-promotion step 2):
#   was iter15_65tkr_reb21_vtg.yaml (leaky env, IR 1.30, turnover 113.6%)
#   now baseline_v5_deploy.yaml     (embargo + lean panel, IR 1.30, turnover 110.3%,
#                                    all 5 primary gates PASS, Haircut Adj SR 0.524)
# Legacy artifacts preserved at outputs/baseline_v4_legacy/.
# To roll back: change this constant to "iter15_65tkr_reb21_vtg.yaml" and
# restore outputs/baseline_v4/ from outputs/baseline_v4_legacy/.
DEFAULT_VARIANT = ROOT / "variants" / "baseline_v5_deploy.yaml"
DEFAULT_RUN_DIR = Path(os.environ.get("CC2_RUN_DIR", ROOT / "outputs" / "baseline_v4"))
DEFAULT_DASHBOARD_PKL = DEFAULT_RUN_DIR / "dashboard_data.pkl"
DEFAULT_DASHBOARD_REPO = Path(
    os.environ.get("CC2_DASHBOARD_REPO", r"C:/Users/westl/Documents/cc2-dashboard")
)
# Anything older than this is flagged in stage_data_check. Generous so weekend
# and holiday gaps don't trigger noise; surface a hard WARN past 7 days.
DATA_AGE_WARN_DAYS = 7

# Files synced to the dashboard repo on every deploy
SYNC_FILES = [
    "streamlit_mobile.py",
    "requirements_dashboard.txt",
]

# Files copied from the variant's output dir to the canonical run_dir after a
# successful --mode full run, so the rest of the pipeline (build_dashboard,
# deploy) sees the new artifacts under the conventional baseline_v4 path.
PROMOTE_FILES = [
    "backtest_result.pkl",
    "metrics.json",
    "experiment_manifest.json",
]


# ---------------------------------------------------------------------------
# Pretty logging
# ---------------------------------------------------------------------------
def _hr(char: str = "=", n: int = 70) -> str:
    return char * n


def step(idx: int, total: int, title: str) -> None:
    print(f"\n{_hr('=')}")
    print(f"  [{idx}/{total}] {title}")
    print(f"{_hr('=')}")


def info(msg: str) -> None:
    print(f"  {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fatal(msg: str, code: int = 1) -> "None":
    print(f"\n  [FATAL] {msg}")
    sys.exit(code)


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
def run(cmd: list[str], cwd: Path | None = None, label: str = "") -> None:
    """Run a command, stream output, raise on non-zero exit."""
    if label:
        info(f"$ {label}")
    info("  " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        fatal(f"command failed (exit={proc.returncode}): {' '.join(cmd)}")
    info(f"  done in {dt:.0f}s")


# ---------------------------------------------------------------------------
# Stage 1 — data sanity (Fix A: freshness check on last data row)
# ---------------------------------------------------------------------------
def _peek_last_data_date(data_file: Path):
    """Return last date in Daily_Returns / PX_LAST sheet without full UniverseData load.

    Reads only column 0 of the candidate sheet via pandas read_excel. Falls
    back through sheet candidates and returns None if all fail (caller logs
    a non-fatal warning).
    """
    import pandas as pd  # local import — keeps module top fast on --help
    for sheet in ("Daily_Returns", "PX_LAST"):
        try:
            df = pd.read_excel(data_file, sheet_name=sheet, usecols=[0])
        except Exception:
            continue
        if df.empty:
            continue
        dates = pd.to_datetime(df.iloc[:, 0], errors="coerce").dropna()
        if len(dates):
            return dates.max(), sheet
    return None, None


def stage_data_check(data_file: Path) -> None:
    """Verify data file exists AND its last row is recent.

    Older versions only checked file existence + mtime. That misses the
    common operator mistake of forgetting to refresh the xlsx — the script
    would happily run a backtest on stale data and push it to production.
    Now we open the workbook and read the last business date in
    Daily_Returns; warn (don't fail) when it's older than DATA_AGE_WARN_DAYS.
    """
    if not data_file.exists():
        fatal(f"data file not found: {data_file}")
    size_mb = data_file.stat().st_size / 1e6
    mtime = datetime.fromtimestamp(data_file.stat().st_mtime)
    info(f"data file: {data_file.name}  ({size_mb:.1f} MB, file modified {mtime:%Y-%m-%d %H:%M})")

    last_date, sheet = _peek_last_data_date(data_file)
    if last_date is None:
        warn("could not determine last data date — Daily_Returns / PX_LAST sheets unreadable")
        return

    import pandas as pd  # already imported by helper, kept local for clarity
    today = pd.Timestamp.today().normalize()
    age_days = int((today - last_date.normalize()).days)
    info(f"data last row ({sheet}): {last_date.date()}  (age: {age_days}d)")
    if age_days <= 1:
        info("data is fresh")
    elif age_days <= DATA_AGE_WARN_DAYS:
        info(f"data is {age_days}d old (acceptable for weekly cadence)")
    else:
        warn(
            f"data is {age_days}d old — last row {last_date.date()}, today {today.date()}.\n"
            f"           Refresh data/{data_file.name} before deploy if you want today's signal."
        )


# ---------------------------------------------------------------------------
# Stage 2 — backtest
# ---------------------------------------------------------------------------
def stage_backtest_incremental(data_file: Path) -> None:
    """daily_update.py — first run does --full-init automatically.

    Note: incremental mode does NOT regenerate baseline_v4/backtest_result.pkl,
    so the dashboard payload built in Stage 3 will reflect the LAST full run,
    not today's incremental update. For dashboard-fresh deploys, prefer
    `--mode full` (the new default).
    """
    state_file = ROOT / "outputs" / "daily_state.pkl"
    if not state_file.exists():
        warn(f"no saved state at {state_file} → bootstrapping with --full-init")
        run(
            [sys.executable, str(ROOT / "daily_update.py"), "--full-init",
             "--data_path", str(data_file)],
            cwd=ROOT,
            label="daily_update.py --full-init",
        )
    else:
        run(
            [sys.executable, str(ROOT / "daily_update.py"),
             "--data_path", str(data_file)],
            cwd=ROOT,
            label="daily_update.py (incremental)",
        )


def _promote_variant_artifacts(variant_dir: Path, run_dir: Path) -> None:
    """Copy variant's pkl/metrics/manifest into the canonical run_dir.

    Fix D: run_variant.py writes to outputs/<label>/, but build_dashboard and
    streamlit_mobile expect outputs/baseline_v4/. Without this copy, --mode
    full would refresh outputs/<label>/ but the dashboard would keep showing
    the previous baseline_v4 result. Variant dir is preserved as audit trail.
    """
    if variant_dir.resolve() == run_dir.resolve():
        return  # already in canonical location
    run_dir.mkdir(parents=True, exist_ok=True)
    for fname in PROMOTE_FILES:
        src = variant_dir / fname
        dst = run_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            info(f"promoted {variant_dir.name}/{fname} → {run_dir.name}/{fname}")
        else:
            warn(f"variant missing {fname} — promote skipped for this file")


def stage_backtest_full(variant: Path, run_dir: Path) -> None:
    """run_variant.py — full walk-forward backtest using a variant yaml.

    After a successful run, promotes pkl/metrics/manifest from the variant
    output dir into the canonical `run_dir` (default outputs/baseline_v4) so
    downstream stages don't have to know variant labels.
    """
    if not variant.exists():
        fatal(f"variant yaml not found: {variant}")
    label = variant.stem  # e.g. "iter15_65tkr_reb21_vtg"
    variant_dir = ROOT / "outputs" / label

    run(
        [sys.executable, str(ROOT / "run_variant.py"), "--variant", str(variant)],
        cwd=ROOT,
        label=f"run_variant.py {variant.name}",
    )

    pkl = variant_dir / "backtest_result.pkl"
    if not pkl.exists():
        fatal(
            f"run_variant.py finished but {pkl} is missing. "
            "Check the run_variant log above for errors."
        )
    info(f"variant artifacts: {variant_dir.relative_to(ROOT)}")
    _promote_variant_artifacts(variant_dir, run_dir)


# ---------------------------------------------------------------------------
# Stage 3 — build dashboard payload
# ---------------------------------------------------------------------------
def stage_build_dashboard(run_dir: Path, data_file: Path, out_pkl: Path) -> None:
    pkl = run_dir / "backtest_result.pkl"
    if not pkl.exists():
        fatal(
            f"backtest_result.pkl not found at {pkl}.\n"
            f"  Did you run --mode full at least once for this run dir?\n"
            f"  Incremental mode does NOT regenerate this pickle."
        )
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, str(ROOT / "scripts" / "build_dashboard_data.py"),
         "--run", str(run_dir),
         "--data", str(data_file),
         "--out", str(out_pkl)],
        cwd=ROOT,
        label="build_dashboard_data.py",
    )
    if not out_pkl.exists():
        fatal(f"build_dashboard_data.py finished but {out_pkl} is missing")
    size_mb = out_pkl.stat().st_size / 1e6
    info(f"dashboard payload: {out_pkl.name} ({size_mb:.2f} MB)")
    if size_mb > 50:
        warn(f"payload {size_mb:.0f} MB may exceed Streamlit Cloud free-tier 1GB RAM budget")


# ---------------------------------------------------------------------------
# Stage 4 — local streamlit smoke
# ---------------------------------------------------------------------------
def stage_smoke() -> None:
    info("starting local streamlit on http://localhost:8501 (Ctrl+C to stop)")
    info("verify: numbers render, last date is today, no errors in console")
    info("then re-run this script with --no-smoke to deploy.")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run",
         str(ROOT / "streamlit_mobile.py")],
        cwd=str(ROOT),
        check=False,
    )


# ---------------------------------------------------------------------------
# Stage 5 — sync & push to dashboard repo
# ---------------------------------------------------------------------------
def _git(args: list[str], cwd: Path, capture: bool = False) -> str:
    cmd = ["git", *args]
    info("$ git " + " ".join(args))
    if capture:
        proc = subprocess.run(cmd, cwd=str(cwd), check=False,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            fatal(f"git failed: {proc.stderr.strip()}")
        return proc.stdout
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        fatal(f"git failed (exit={proc.returncode}): git {' '.join(args)}")
    return ""


def stage_deploy(dashboard_repo: Path, run_dir: Path,
                 dashboard_pkl: Path, push: bool) -> None:
    if not dashboard_repo.exists():
        fatal(
            f"dashboard repo not found: {dashboard_repo}\n"
            f"  set CC2_DASHBOARD_REPO env var or pass --dashboard-repo"
        )
    if not (dashboard_repo / ".git").exists():
        fatal(f"{dashboard_repo} is not a git repo")

    # Sync code files
    for fname in SYNC_FILES:
        src = ROOT / fname
        dst = dashboard_repo / fname
        if not src.exists():
            warn(f"{fname} not found at source; skipping")
            continue
        shutil.copy2(src, dst)
        info(f"copied {fname}")

    # Sync dashboard payload (preserve relative path under outputs/)
    rel = dashboard_pkl.relative_to(ROOT)
    dst_pkl = dashboard_repo / rel
    dst_pkl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dashboard_pkl, dst_pkl)
    info(f"copied {rel.as_posix()}  ({dst_pkl.stat().st_size/1e6:.2f} MB)")

    # Git
    status = _git(["status", "--porcelain"], cwd=dashboard_repo, capture=True)
    if not status.strip():
        info("no changes in dashboard repo — nothing to commit")
        return

    _git(["add", *SYNC_FILES], cwd=dashboard_repo)
    # -f because dashboard_data.pkl is in .gitignore
    _git(["add", "-f", str(rel.as_posix())], cwd=dashboard_repo)

    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"update: dashboard data {today}"
    _git(["commit", "-m", msg], cwd=dashboard_repo)

    if push:
        _git(["push", "origin", "main"], cwd=dashboard_repo)
        info("pushed → Streamlit Cloud will auto-redeploy in ~30s")
        info("check: https://share.streamlit.io  or your app URL")
    else:
        info("--no-push set; commit created locally but not pushed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="One-shot: data → backtest → dashboard build → deploy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Fix B(2): default mode is 'full' so the bat one-click really does refresh
    # baseline_v4/backtest_result.pkl (and therefore the dashboard).
    p.add_argument("--mode", choices=["incremental", "full"], default="full",
                   help="full (default): run_variant.py + auto-promote to baseline_v4. "
                        "incremental: daily_update.py (fast, but does NOT refresh backtest pkl).")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE,
                   help=f"data xlsx (default: {DEFAULT_DATA_FILE.name})")
    p.add_argument("--variant", type=Path, default=DEFAULT_VARIANT,
                   help=f"variant yaml for --mode full (default: {DEFAULT_VARIANT.name} = production)")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                   help=f"canonical backtest output dir (promote target). "
                        f"(default: {DEFAULT_RUN_DIR.relative_to(ROOT) if DEFAULT_RUN_DIR.is_relative_to(ROOT) else DEFAULT_RUN_DIR})")
    p.add_argument("--dashboard-repo", type=Path, default=DEFAULT_DASHBOARD_REPO,
                   help="local clone of cc2-dashboard")
    p.add_argument("--no-push", action="store_true",
                   help="commit locally but don't push to GitHub")
    p.add_argument("--no-build", action="store_true",
                   help="skip dashboard payload build (assumes pkl already exists)")
    p.add_argument("--no-deploy", action="store_true",
                   help="skip git sync/commit/push entirely")
    p.add_argument("--smoke", action="store_true",
                   help="run local streamlit after build (blocks until Ctrl+C)")
    args = p.parse_args()

    dashboard_pkl = args.run_dir / "dashboard_data.pkl"

    print(_hr("="))
    print("  AI Signal — Update & Deploy")
    print(f"  mode={args.mode}  variant={args.variant.name}")
    print(f"  run_dir={args.run_dir}")
    print(f"  dashboard_repo={args.dashboard_repo}")
    print(_hr("="))

    t_start = time.time()

    # 1. data check
    step(1, 5, "Data sanity check")
    stage_data_check(args.data)

    # 2. backtest
    step(2, 5, f"Backtest ({args.mode})")
    if args.mode == "incremental":
        stage_backtest_incremental(args.data)
    else:
        stage_backtest_full(args.variant, args.run_dir)

    # 3. dashboard payload
    step(3, 5, "Build dashboard payload")
    if args.no_build:
        info("--no-build set; skipping")
        if not dashboard_pkl.exists():
            fatal(f"--no-build requires existing {dashboard_pkl}")
    else:
        stage_build_dashboard(args.run_dir, args.data, dashboard_pkl)

    # 4. smoke
    step(4, 5, "Local smoke (optional)")
    if args.smoke:
        stage_smoke()
    else:
        info("--smoke not set; skipping local preview")

    # 5. deploy
    step(5, 5, "Deploy to cc2-dashboard")
    if args.no_deploy:
        info("--no-deploy set; skipping")
    else:
        stage_deploy(args.dashboard_repo, args.run_dir,
                     dashboard_pkl, push=not args.no_push)

    # final-v1-promotion step 4: live-vs-backtest drift monitor.
    # Non-fatal — deploy proceeds regardless.
    try:
        run([sys.executable, str(ROOT / "scripts" / "compute_live_delta.py")],
            cwd=ROOT, label="compute_live_delta.py")
    except SystemExit:
        warn("live-delta monitor failed; deploy continues (monitor is advisory only)")

    dt = time.time() - t_start
    print(f"\n{_hr('=')}")
    print(f"  ALL DONE in {dt:.0f}s")
    print(f"{_hr('=')}")


if __name__ == "__main__":
    main()
