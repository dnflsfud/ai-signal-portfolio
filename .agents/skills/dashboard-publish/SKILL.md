---
name: dashboard-publish
description: "백테스트 결과를 CSV로 내보내고 Streamlit 대시보드를 실행한다. 'CSV 내보내기', '대시보드 실행', 'Streamlit 실행', '결과 저장', '리포트 생성' 요청 시 이 스킬을 사용."
---

# Dashboard Publish Skill

BacktestResult를 dashboard payload로 변환하고 Streamlit 대시보드를 실행한다.

운영 워크플로의 정식 entry-point는 `update_and_deploy.bat` (root). 본 skill은 중간 단계 직접 호출용.

## 워크플로우

### Step 1: Dashboard payload 빌드
```bash
python scripts/build_dashboard_data.py \
    --run outputs/baseline_v4 \
    --data data/ai_signal_data.xlsx \
    --out outputs/baseline_v4/dashboard_data.pkl
```
출력: `outputs/baseline_v4/dashboard_data.pkl` (~3MB; 65MB backtest_result.pkl을 압축한 IC table, feature importance, group PnL, score breakdowns 포함).

### Step 2: Streamlit 대시보드 (로컬 smoke)
```bash
python -m streamlit run streamlit_mobile.py --server.port 8501 --server.address 127.0.0.1
```

### Step 3: cc2-dashboard repo로 배포 (production)
`update_and_deploy.py --no-build` 또는 `update_and_deploy.bat`:
- `streamlit_mobile.py` + `requirements_dashboard.txt` + `dashboard_data.pkl`을 cc2-dashboard repo로 sync
- `git add/commit/push origin main` → Streamlit Cloud가 ~30초 후 자동 재배포
- force-push 금지, `--no-push` 옵션으로 로컬 commit만도 가능

## TICKER_META 사용
```python
from src.metadata import TICKER_META, TICKER_SECTOR
# TICKER_META: {ticker: {"sector": ..., "style": ..., "sub": ...}}
# TICKER_SECTOR: {ticker: sector_string}
```
인라인 정의 금지 — 반드시 src/metadata.py에서 import.
