#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py - One-Shot Pipeline
===============================
데이터 로드 -> 피처 생성 -> 모델 학습 -> 백테스트 -> Attribution -> CSV 내보내기 -> GitHub Push -> Streamlit 실행

사용법:
  python run_all.py                    # 전체 파이프라인 + Streamlit 실행
  python run_all.py --no-streamlit     # 파이프라인만 (Streamlit 스킵)
  python run_all.py --streamlit-only   # CSV가 이미 있으면 Streamlit만 실행
"""

import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_pipeline():
    """main.py 실행 (전체 파이프라인)."""
    print("=" * 70)
    print("  AI Signal Portfolio - Full Pipeline")
    print("  Config A: Penalty=0.3, MaxTO=0.15, 10-day Rebalancing, MCW BM")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"),
         "--data_path", str(PROJECT_ROOT / "data" / "ai_signal_data.xlsx"),
         "--output_dir", str(PROJECT_ROOT / "outputs")],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("\n[ERROR] Pipeline failed!")
        return False

    print("\n[SUCCESS] Pipeline completed! CSVs generated and pushed to GitHub.")
    return True


def run_streamlit():
    """Streamlit 대시보드 실행."""
    app_path = PROJECT_ROOT / "app.py"
    if not app_path.exists():
        print(f"[ERROR] {app_path} not found!")
        return False

    csv_dir = PROJECT_ROOT / "outputs" / "csv"
    if not (csv_dir / "daily_performance.csv").exists():
        print(f"[ERROR] CSV files not found in {csv_dir}.")
        print("  Run pipeline first: python run_all.py")
        return False

    print("\n" + "=" * 70)
    print("  Launching Streamlit Dashboard...")
    print(f"  http://localhost:8501")
    print("=" * 70)

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", "8501"],
        cwd=str(PROJECT_ROOT),
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="AI Signal Portfolio - One-Shot Runner")
    parser.add_argument("--no-streamlit", action="store_true", help="Pipeline only (skip Streamlit)")
    parser.add_argument("--streamlit-only", action="store_true", help="Streamlit only (skip pipeline)")
    args = parser.parse_args()

    if args.streamlit_only:
        run_streamlit()
    elif args.no_streamlit:
        run_pipeline()
    else:
        success = run_pipeline()
        if success:
            run_streamlit()


if __name__ == "__main__":
    main()
