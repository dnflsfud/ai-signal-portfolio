---
name: data-pipeline
description: "AI Signal 포트폴리오의 데이터 로드, 피처 엔지니어링, PCA 타겟 생성을 실행한다. '데이터 준비', '피처 생성', '타겟 생성', 'prepare_data 실행', 'Phase 1~3 실행' 요청 시 이 스킬을 사용."
---

# Data Pipeline Skill

AI Signal 포트폴리오의 Phase 1~3 데이터 파이프라인을 실행한다.

## 워크플로우

### Step 1: 데이터 로드 (Phase 1)
```python
from src.data_loader import UniverseData
data = UniverseData("./data/ai_signal_data.xlsx")
```
- 22개 시트 로드, 날짜 인덱스 통일
- Sent_Trend 회사명 → 티커 매핑
- 결측치: ffill → cross-sectional median

### Step 2: 피처 엔지니어링 (Phase 2)
```python
from src.feature_engine import build_all_features
panel, feature_names, feature_groups = build_all_features(data)
```
- ~350개 피처 생성 (Accounting, Price, Sellside, Conditioning, Factor)
- Cross-sectional Z-score 정규화
- +-5 std 클리핑

### Step 3: 타겟 생성 (Phase 3)
```python
from src.target_engine import build_targets
targets = build_targets(data, n_remove=2)  # Partial PCA: PC1+PC2만 제거
```
- 20일 Specific Return = PCA 잔차
- n_remove=2: 시장+사이즈만 제거, 섹터 알파 보존

### Step 4: 검증
- 피처 수 >= 300 확인
- 타겟 유효 관측치 비율 > 50% 확인
- 날짜 범위 및 종목 수 확인

### Step 5: 저장
```python
from src.backtest import save_checkpoint
save_checkpoint("phase1", {"data": data})
save_checkpoint("phase2", {"panel": panel, "feature_names": feature_names, "feature_groups": feature_groups})
save_checkpoint("phase3", {"targets": targets})
```

## PipelineConfig 파라미터
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| pca_components | 5 | PCA 성분 수 |
| pca_n_remove | 2 | 제거 성분 수 |
| pca_lookback | 252 | PCA fitting 기간 |
| forward_horizon | 20 | Forward return 기간 |

## 검증 기준
| 항목 | 임계값 | 실패 시 |
|------|-------|--------|
| 시트 수 | >= 15 | 중단 |
| 영업일 수 | >= 2000 | 경고 |
| 종목 수 | >= 10 | 중단 |
| 피처 수 | >= 300 | 경고 |
| 타겟 유효율 | > 50% | 경고 |
