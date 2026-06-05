# AI Signal Portfolio Construction System

## 프로젝트 개요
Gradient Boosting 기반 비선형 시그널 결합 + Mean-Variance Optimization을 통한 포트폴리오 구축 시스템

## 핵심 설계 원칙
1. ML 모델은 20일 Specific Return(잔차)을 예측한다 - 전체 수익률이 아님
2. 입력 피처는 전통적(경제적 근거 있음) + conditioning 변수의 조합
3. 피처 결합만 비선형(Gradient Boosting), 개별 피처 구축은 전통적 방식
4. 포트폴리오 구축은 전통적 MVO - ML은 리턴 예측에만 사용
5. Turnover penalty로 단기 모델의 회전율을 제어

## 기술 스택
- Python 3.10+
- pandas, numpy, scipy, sklearn
- LightGBM (Gradient Boosting)
- cvxpy (Mean-Variance Optimization)
- shap (모델 해석)
- matplotlib, plotly (시각화)

## 디렉토리 구조
ai_signal_cc2_harness/
├── AGENTS.md            # (본 문서) 설계 의도 + config 스냅샷
├── CLAUDE.md            # AGENTS.md 와 동일 사본 (Claude Code 진입점)
├── data/
│   ├── ai_signal_data.xlsx    # production 데이터 (config.py 기본값)
│   └── RL_Universe_Data.xlsx  # 초기 ~15종목 데이터 (legacy, 단발 실험용)
├── src/
│   ├── config.py        # SSOT — DEFAULT_CONFIG dataclass
│   ├── data_loader.py
│   ├── feature_engine.py
│   ├── target_engine.py
│   ├── model_trainer.py
│   ├── portfolio_optimizer.py
│   ├── attribution.py
│   ├── backtest.py
│   ├── harness.py       # variant override + sub-period IR
│   └── utils.py
├── variants/            # YAML variant manifests for run_variant.py
├── scripts/             # 보조 스크립트 (build_dashboard_data.py)
├── outputs/
│   ├── baseline_v4/                  # 현 production (canonical run_dir)
│   ├── iter15_65tkr_reb21_vtg/       # production variant 원본 (promote 전)
│   ├── csv/                          # daily_update 산출 CSV
│   └── reports/
├── update_and_deploy.bat / .py   # 운영 entry-point (data check → backtest → dashboard build → deploy)
├── run_variant.py       # YAML manifest로 variant 실행 (Stage 2 full backend)
├── daily_update.py      # 증분 일간 업데이트 (Stage 2 incremental backend)
├── streamlit_mobile.py  # cc2-dashboard repo로 sync되는 production 대시보드
└── run_selection_bias.py  # Deflated Sharpe Ratio 검증 (Bailey & Lopez de Prado 2014)

## 데이터 사양

### 원본 파일: data/ai_signal_data.xlsx (production, config.py `data_path` 기본값)
- 기간: 2014.01 ~ 2026.04 (약 3,200 영업일 / panel 기준)
- 종목: 50~65개 (production 유니버스, REDESIGN I 이후 essential-sheet 교집합으로 JPM/GS 등 금융주 포함)
- 초기 ~15종목 시안은 `data/RL_Universe_Data.xlsx` 에 보존 (legacy)
- 정확한 ticker 리스트는 `src/data_loader.TICKERS` 참조

#### Initial 15종목 (legacy 시안)
AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, PLTR, AVGO, MU, GEV, VRT, BE, LITE, 000660 (SK Hynix)

### 시트 -> 피처 카테고리 매핑

| 시트명 | 용도 | 피처 카테고리 |
|--------|------|---------------|
| PX_LAST | 종가 | Price 기반 피처 |
| Daily_Returns | 일간 수익률 | 타겟 및 Price 피처 |
| BEST_EPS | 컨센서스 EPS | Accounting |
| BEST_SALES | 컨센서스 매출 | Accounting |
| BEST_PE_RATIO | PE | Valuation |
| BEST_PEG_RATIO | PEG | Valuation |
| BEST_CALCULATED_FCF | FCF | Accounting |
| BEST_GROSS_MARGIN | 매출총이익률 | Accounting |
| CUR_MKT_CAP | 시가총액 | Size/Conditioning |
| OPER_MARGIN | 영업이익률 | Accounting |
| BEST_CAPEX | CAPEX | Accounting |
| BEST_ROE | ROE | Accounting |
| BEST_PX_BPS_RATIO | PBR | Valuation |
| BEST_EV_TO_BEST_EBITDA | EV/EBITDA | Valuation |
| NEWS_SENTIMENT_DAILY_AVG | 뉴스 센티먼트 | Sentiment |
| EQY_REC_CONS | 애널리스트 추천 | Sell-side |
| Sent_Trend_Momentum_Timeseries | 센티먼트 모멘텀 | Sentiment |
| Sent_Trend_21d_Timeseries | 21일 센티먼트 트렌드 | Sentiment |
| Factset_EPS_Revision | EPS 리비전 | Sell-side |
| Factset_Sales_Revision | 매출 리비전 | Sell-side |
| Factset_TG_Price | 목표주가 | Sell-side |
| Universe_Meta | 종목 메타 (섹터) | Conditioning |

주의: Sent_Trend 시트 컬럼명이 회사명이므로 티커 매핑 필요.
매핑: Apple->AAPL, Microsoft->MSFT, Alphabet->GOOGL, Amazon->AMZN, Meta->META, Nvidia->NVDA, Tesla->TSLA, Palantir->PLTR, Broadcom->AVGO, Micron->MU, GE Vernova->GEV, Vertiv->VRT, Bloom Energy->BE, Lumentum->LITE, SK Hynix->000660
주의: 000660은 KRW 표시이므로 수익률 기반으로 통일.
결측치: ffill -> 남은 NaN은 해당 날짜 cross-sectional median.

---

## 상세 구현 사양

### Phase 1: 데이터 로드 및 전처리 (src/data_loader.py)
- 모든 시트를 pandas DataFrame으로 로드
- 날짜 인덱스 통일 (BusinessDays 기준)
- Sent_Trend 시트 회사명을 티커로 매핑
- 결측치 처리: ffill -> cross-sectional median

### Phase 2: 피처 엔지니어링 (src/feature_engine.py)
각 피처는 날짜별 cross-sectional Z-score로 정규화.

#### 카테고리 1: Accounting/Fundamental 변화 (~25%)
각 지표(EPS, SALES, FCF, GROSS_MARGIN, OPER_MARGIN, CAPEX, ROE)에 대해:
- 단기 변화율: pct_change(5), pct_change(10), pct_change(21)
- 장기 변화율: pct_change(63), pct_change(126), pct_change(252)
- 변화 가속도: chg_21d - chg_63d
- Level Z-score: cross-sectional zscore of raw level

#### 카테고리 2: Price/Market 기반 (~25%)
- Reversal: -1 * rolling_sum(returns, w) for w in [5, 10, 21]
- Momentum: rolling_sum(returns, w) for w in [63, 126, 252]
- Risk-adjusted momentum: momentum / rolling_std
- Realized volatility: rolling_std * sqrt(252) for w in [21, 63, 126]
- Volatility ratio: vol_21d / vol_126d
- Price distance from MA: (price / MA) - 1 for MA in [21, 50, 200]
- Drawdown: price / rolling_max(63) - 1
- Market cap rank: cross-sectional percentile

#### 카테고리 3: Sell-side/Sentiment (~25%)
- Analyst recommendation level, diff(21d), diff(63d)
- Target price upside: (tg_price / px_last) - 1, diff(21d)
- EPS/Sales revision score, diff(21d), rolling_mean(63d)
- News sentiment: raw, MA(5), MA(21), trend(MA5-MA21)
- Sentiment trend momentum, 21d trend

#### 카테고리 4: Conditioning 변수 (~10%)
- Calendar: month, day_of_month, day_of_week, week_of_year, quarter
- is_month_end_week, is_quarter_end, is_january
- Earnings season proxy: 1-2월(Q4), 4-5월(Q1), 7-8월(Q2), 10-11월(Q3)
- Sector one-hot encoding
- Market regime: 21d EW return, 63d cross-sectional avg vol
- Size bucket: is_mega_cap(rank>0.8), is_small(rank<0.3)

최종 피처 수 (설계 시안): 약 80~120개.
**실측 (production deploy, feature_mode="lean" + EWMA prune; 2026-05-20 promotion).**
참고용 `core` 모드 panel은 61개 / 7 그룹 (아래 *Feature 구성* 표).

### Phase 3: 타겟 변수 (src/target_engine.py)
20일 Specific Return = PCA 잔차 수익률.

각 시점 t에서:
1. 과거 252일 일간 수익률로 PCA fitting (n_components=5)
2. t~t+20 영업일 forward cumulative return
3. PCA common component 제거
4. 잔차 = Specific Return = 타겟

look-ahead bias 방지: PCA fitting은 반드시 과거 데이터만.

### Phase 4: LightGBM 모델 (src/model_trainer.py)
- objective: regression (연속값)
- 출력: cross-sectional Z-score -> expected_return 변환
- 훈련: 3년(756일) rolling window
- 재훈련: 3개월(63일)마다
- Validation: 훈련 마지막 6개월
- EWMA feature importance (alpha=0.3) 로 하위 5% 피처 drop, 최소 60개 유지

LightGBM params (src/config.py `lgbm_params` – 실제 실행값, 2026-04-13 V2 패턴):
  learning_rate=0.02, num_leaves=31, max_depth=5,
  min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
  reg_alpha=0.3, reg_lambda=2.0, n_estimators=800,
  early_stopping_rounds=100, random_state=42

> 원 설계(Pictet 기준)는 lr=0.05 / leaves=31 / depth=6 / n_est=500 이었으나,
> REDESIGN D 튜닝을 거쳐 현재값으로 수렴 (이전 lr=0.008/leaves=63/depth=7/n_est=1500
> 설정은 degenerate 모델 → 안정 수렴 위해 V2 패턴으로 재조정).
> 재현은 항상 src/config.py `DEFAULT_CONFIG.lgbm_params` 를 기준.

### Phase 5: Walk-Forward 백테스트 (src/backtest.py)
- train_window: 1260일 (5년, config.py 실제값 — 다중 레짐 커버리지)
- retrain_freq: 63일 (3개월)
- prediction_horizon: 20일
- rebalance_freq: 21일 (월간, config.py 실제값 — REDESIGN R)
- one_way_tc: 10bps

> 원 설계는 주간(5일) 리밸런싱 / 3년(756일) 윈도우였으나,
> turnover 제어 + 다중 레짐 커버를 위해 월간 21일 / 5년 윈도우로 조정 (iter6 baseline turnover 455% → ~225% 감소).

### Phase 6: 포트폴리오 최적화 (src/portfolio_optimizer.py)
cvxpy Mean-Variance Optimization.

목적함수: Maximize(E[r] @ w - lambda * risk - tc * turnover)

config.py 실제값 (REDESIGN E + iter9 baseline):
- risk_aversion = 1.0 (iter9 baseline — risk_aversion 0.5는 TE quad 제약 때문에 binding 안 되어 효과 無)
- turnover_penalty = 0.03 (Pictet 0.005 대비 6배 — 적정 보수 모드)
- max_te_annual = 0.045 (REDESIGN P — codex_v2 baseline 수준 4.5%로 alpha 공간 확보)
- max_single_turnover = 0.15
- cov_lookback = 126일 (Ledoit-Wolf shrinkage)
- bm_weight_floor = 0.02 (UW 공간 확보)
- max_active_share = 0.50 (post_init: portfolio_style="core_satellite" 시 2 × satellite_budget = 0.45로 자동 축소)

제약 (config.py 실제값):
- sum(w) = 1, w >= 0
- w <= max_weight = 0.15 (15%)
- |w - bm_w| <= max_active_per_stock = 0.12
  (단, portfolio_style="core_satellite" 시 satellite_max_per_stock = 0.04로 자동 축소)
- 섹터 편차 <= sector_deviation = ±0.10 (cap-weighted benchmark 기준)

벤치마크: cap-weighted (CUR_MKT_CAP, REDESIGN A — EW 1/n에서 변경)

> 원 설계는 EW 1/n 벤치마크 / Pictet turnover_penalty 0.005 였음.
> EW는 mega-cap 비중 ~2%로 묶여 NVDA/MSFT 등에서 active room이 거의 없음 → cap-weighted로 전환.
> REDESIGN E에서 이전 turnover_penalty 0.3 (Pictet 60배 보수)을 0.03으로 풀어 신호 반영도 회복.

### Phase 7: Attribution (src/attribution.py)
- SHAP TreeExplainer
- Feature group별 기여도 (Accounting, Price, Sellside, Conditioning)
- 선형 vs 비선형 분해 (목표 ~50/50)

### Phase 8: 시각화 (run_variant.py + scripts/build_dashboard_data.py)
outputs/baseline_v4/ + outputs/csv/ 에 저장. 휴대폰 대시보드는 `streamlit_mobile.py`가 `dashboard_data.pkl`만 read.
1. 누적 수익률 (전략 vs 벤치마크)
2. Rolling IR (252일)
3. Drawdown
4. 월별 수익률 히트맵
5. SHAP feature importance
6. Feature group 기여도 시계열
7. 선형/비선형 비율 추이
8. IC 시계열
9. 포트폴리오 비중
10. 재훈련 전후 상관 추이

## 실행 방법

```bash
# 의존성 (requirements.txt 기준)
pip install -r requirements.txt
# 또는: pip install pandas numpy scipy scikit-learn lightgbm cvxpy shap matplotlib plotly openpyxl pyyaml

# 1) 운영 entry-point (전체 흐름: backtest → dashboard build → cc2-dashboard repo로 push)
update_and_deploy.bat                              # 더블클릭 = full mode (~3-4분)
update_and_deploy.bat --mode incremental           # 가벼운 일간 갱신 (~30초~1분)
# 상세: docs/UPDATE_AND_DEPLOY_FLOW.md

# 2) Variant만 실행 (production = baseline_v5_deploy, 2026-05-19 v2 cutover)
python run_variant.py --variant variants/baseline_v5_deploy.yaml
# Legacy (rollback용): python run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml

# 3) 일간 증분 업데이트 단독 호출 (update_and_deploy.bat --mode incremental와 동등)
python daily_update.py --full-init    # 첫 실행 (전체 백테스트 + state 저장)
python daily_update.py                # 이후: 새 가격만 처리

# 4) Dashboard payload 빌드 (Stage 3 단독 호출)
python scripts/build_dashboard_data.py --run outputs/baseline_v4 --data data/ai_signal_data.xlsx
```

## 핵심 파라미터 (src/config.py `DEFAULT_CONFIG` 기준 – 실제 실행값)

### 프로덕션 구성 = **Lean (feature_mode) + Core-Satellite + Score-Gate + Cap-Weighted BM** (2026-05-20 baseline_v5 promotion ~ )

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| **feature_mode** | **lean** | overlay-ablation Task B에서 단일 knob 최대 IR 개선 (+0.506 trimmed) → `DEFAULT_CONFIG.feature_mode` 가 2026-05-20에 lean으로 promotion됨. 참고용 core panel(61개 / 7 그룹)은 아래 표 |
| **benchmark_type** | **cap_weighted** | 시가총액 가중 (REDESIGN A) |
| **portfolio_style** | **core_satellite** | ~78% 코어 (BM 추적) + ~22% 새틀라이트 (active) |
| satellite_budget | **0.225** | 일방향 active share 목표 (= L1/2) — iter9 baseline |
| satellite_max_per_stock | **0.04** | 종목당 active tilt 상한 (±4%) — iter19 single-stock 리스크 제한 |
| **enforce_score_gated_ow** | **True** | 모든 OW는 모델 신호 z>0 필요 — MVO diversification OW 차단 |
| score_threshold_for_ow | 0.0 | 긍정 z-score 하한 (엄격화 원하면 +0.25~+0.50) |
| max_active_share (L1) | **0.50** → post_init 후 **0.45** | = 2 × satellite_budget로 자동 축소 (`__post_init__`) |
| max_active_per_stock | **0.12** → post_init 후 **0.04** | satellite_max_per_stock에 의해 자동 축소 |
| max_weight | **0.15** | 종목당 절대 weight 상한 |
| bm_weight_floor | **0.02** | bm의 2% 최소 유지 (UW 공간 확보) |
| risk_aversion | **1.0** | iter9 baseline |
| turnover_penalty | **0.03** | Pictet 원안(0.005)의 6배 (이전 0.3 대비 1/10로 완화) |
| sector_deviation | **±0.10** | 섹터 중립에 가깝게 |
| max_te_annual | **0.045** | REDESIGN P — codex_v2 baseline 4.5% |
| no_trade_band | 0.003 | 30bp 미만 변화는 트레이드 스킵 (REDESIGN J) |
| partial_rebalance_eta | 0.50 | 1회 리밸런싱당 목표 변화량 50% 만 집행 |
| prediction_ema_alpha | **0.5** | 신호 EMA — REDESIGN R-9 (P2 회복: 0.8→0.5) |
| 예측 타겟 | 20일 Specific Return | Pictet 기준 |
| PCA 성분 수 / 제거 | 5 / 2 | partial PCA residual (REDESIGN L) |
| train_window | **1260일 (5년)** | 다중 레짐 커버리지 |
| retrain_freq | 63일 (분기) | Pictet 기준 |
| rebalance_freq | **21일 (월간)** | REDESIGN R — turnover 제어 (이전 10일에서 변경) |
| one_way_tc | 10 bps | 모든 종목 공통 편도 TC (스칼라) |
| **fx_surcharge_per_ticker** | **{000660: 3bp, 005930: 3bp}** | **fx-cost-modeling (2026-05-21) — KRW↔USD spot bid-ask + slippage; one_way_tc 위에 가산. round-trip 6bp 추가** |
| **embargo_days** | **20** | **data-leakage-fix (2026-05) — walk-forward train/val/predict 간 라벨 누수 차단** |
| **enforce_oos_holdout** | **True** | **data-leakage-fix (2026-05) — research 모드에서 자동 적용** |
| **train_cutoff_date** | **"2024-12-31"** | **data-leakage-fix (2026-05) — research 모드는 이 날짜까지만 학습/예측** |

### Post-prediction 조정 모듈 (production ON)

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| pead_boost_enabled | True | Post-Earnings Announcement Drift 보너스 (decay 7일, 21일 cutoff) |
| pead_boost_weight | 0.30 | 최대 boost (z-score 단위) |
| growth_tilt_enabled | True | 성장/리비전 tilt (rev 50% / fundamental 50%, EPS:Sales 50:50) |
| growth_tilt_weight | 0.25 | tilt boost (z-score 단위) |
| mega_cap_protection_enabled | True | mega-cap (bm≥4%) 비대칭 active 제약 |
| mega_cap_funding_mode | True | UW를 K=4개 worst-scoring mega-cap에 집중 |

### Optional / OFF (default = False — 검증 후 promote 후보)

| 파라미터 | 의미 |
|----------|------|
| regime_aware_pca_lookback | vol regime 기반 PCA lookback 동적 전환 |
| regime_pca_weighted_enabled | regime conditional 가중 PCA fit |
| multi_horizon_targets_enabled | 5d/20d/63d 멀티 horizon 타겟 앙상블 |
| bm_proportional_cap_enabled | BM/vol 비례 active cap (mega_cap_protection 일반화) |
| signal_stability_lambda | retrain간 score 변화 shrinkage |
| value_trap_gate_enabled | (단, `baseline_v5_deploy` / `baseline_v5` / legacy `iter15_65tkr_reb21_vtg` 에선 ON으로 override; `vtg_scale=0.0`이라 실효 비활성) |
| enforce_oos_holdout | 튜닝 시 train_cutoff_date 이후 데이터 차단 |

> **SSOT**: `src/config.py` 의 `DEFAULT_CONFIG`. 본 문서는 그 스냅샷이며, 충돌 시 config.py가 진실.
> 실행 시마다 `outputs/experiment_manifest.json` 에 현재 config + git hash 스냅샷 저장.
> Variant 실행은 `variants/<label>.yaml` 의 `overrides:` 가 DEFAULT_CONFIG 위에 적용됨 (`run_variant.py`).

### Feature 구성 — 참고용 `core` 모드 panel 61개 / 7 그룹 (실측: outputs/csv/feature_importance.csv 기준)

> 2026-05-20 promotion 이후 production은 `feature_mode="lean"`. 아래 표는 직전 `core` 모드 구성의 스냅샷이며, lean panel 의 실 분포는 `outputs/csv/feature_importance.csv` 참조.

| 그룹 | 개수 | 비중 | 비고 |
|---|---|---|---|
| **Accounting** (Quality/Growth/Value) | 18 | 29.5% | oper_margin_chg_63d, cash_conversion_z, best_peg_ratio_level_z |
| **Price** (Momentum/Risk) | 12 | 19.7% | momentum_252d, beta_63d, ma_cross_50_200, realized_vol_126d |
| **Financials** (BS/CF) | 11 | 18.0% | REDESIGN I — JPM/GS 복원 시 추가된 essential-sheet 그룹 |
| **Sellside** (Analyst/Revision) | 8 | 13.1% | analyst_rec_level, eps_rev_ma_63d, tg_mom_63d |
| **Factor** (Macro) | 5 | 8.2% | fac_yield_slope, fac_F_Quality_mom_63d, fac_value_growth_63d |
| **MacroCross** (interaction) | 5 | 8.2% | rate×rev, slope×rev, VIX×mom252, vol×mom63, DXY×rev (Phase 2, 2026-04-22) |
| **Conditioning** (Regime) | 2 | 3.3% | earn_cycle_pos, regime_mkt_ret_21d |

**합계: 61 features / 7 groups.**

> 이전 핸드픽 46개 (Accounting 18 / Price 12 / Sellside 8 / Factor 5 / Conditioning 3) 디자인에
> Phase 2 MacroCross 5개 + Financials 11개가 추가되어 위 `core` 모드는 61개. 현재 default 인
> `feature_mode="lean"` panel 은 `src/feature_engine.build_all_features()` + EWMA prune 결과로 산출되며,
> 정확한 리스트는 `outputs/csv/feature_importance.csv` 참조.

### 최종 성과 — 세 baseline 분리 노출 (2026-05-19 v2 cutover 이후)

> 1) **research baseline** (canonical gate denominator, conservative anchor):
>    `iter15_FINAL_postfix` — `outputs/iter15_FINAL_postfix/` + `comparison.md` (2026-05-19).
>    `tuning_mode: research`, cutoff=2024-12-31, embargo=20. 윈도우 1592일.
>    **신규 변형 promotion gate 의 분모로만 사용** (`docs/BASELINE.md` Gate 1-5).
>
> 2) **production deploy baseline** (현재 `update_and_deploy.bat` 진입점):
>    `baseline_v5_deploy` — `outputs/baseline_v5_deploy/` (+ alias `outputs/baseline_v4/`).
>    `tuning_mode: deploy`, cutoff OFF. final-v1-promotion (2026-05-19 v2) 에서 cutover.
>    feature_mode=lean 단일 knob 적용 (overlay-ablation Task B 결과).
>
> 3) **legacy deploy baseline** (archived, 누수 환경):
>    `iter15_65tkr_reb21_vtg` — `outputs/baseline_v4_legacy/`. 2026-04-24 ~ 2026-05-19 production.
>    audit/rollback 용 보존.

| 지표 | 1) research baseline (gate denom.) | 2) production deploy (current) | 3) legacy deploy (archived) |
|---|---:|---:|---:|
| 매니페스트 | `iter15_FINAL_postfix.yaml` | `baseline_v5_deploy.yaml` | `iter15_65tkr_reb21_vtg.yaml` |
| 윈도우 | 2018-11 → 2024-12 (trimmed) | 2018-11 → 2026-05 (full) | 2018-11 → 2026-05 (full, leaky) |
| Annual Return | 24.58% | **30.17%** | 29.96% |
| Active Return / yr | +1.26% | **+4.40%** | +4.10% |
| Tracking Error | 2.88% | **3.12%** | 3.14% |
| Sharpe | 1.094 | **1.367** | 1.332 |
| **Information Ratio** | **0.392** | **1.408** | 1.304 |
| Max Drawdown | −29.95% | −29.34% | −30.34% |
| Annual Turnover 2-way | 90.8% | **102.4%** | 113.6% |
| Avg IC | 0.0463 | **0.0478** | 0.0450 |
| P1 IR (2018-11~2021-05) | +1.287 | +0.667 | +0.844 |
| P2 IR (2021-05~2023-10) | **−0.497** | **+1.515** | +0.783 |
| P3 IR (2023-10~) | +0.390 (trim) | **+1.740** | +2.041 |
| rolling_ir_pos_frac | n/a | **0.801** | 0.818 |
| SPA p-value | n/a | **0.000** | 0.000 |
| 유니버스 | 65 | 65 | 65 |

> **해석**: legacy(3) IR 1.30 → 동일 cutoff-trimmed 윈도우 IR=0.80 → embargo 추가 시 0.39
> ⇒ 누수 프리미엄 ΔIR=−0.41 였음. 이 누수를 제거하면 P2가 양수→음수로 뒤집힌다.
> production deploy(2) 는 same lean panel 을 cutoff OFF 환경 (legacy 동일) 에서 돌린 것 →
> headline IR=1.41 (audit-fix re-sim, 2026-06-05) 로 legacy(1.30)를 상회하며, P1 weak / P2 strong 으로 regime mix 가
> 건강해졌고 (P2 +1.52 vs legacy +0.78), turnover 도 legacy 대비 −11.2pp 개선됐다.
> production deploy(2) 수치는 audit 수정(MVO-1 constraint-preserving fallback + DYNEXEC-1 sharpness 복원,
> commit `b612d75`)을 캐시된 Phase 1-4 예측 위에 Phase 5-6 재시뮬레이션한 결과 (window = 2018-11→2026-05-15).
> 자세한 lineage 는 `docs/BASELINE.md` 참조.

### Selection Bias 검증 — baseline_v5 (recount, 2026-05-19 v2 — CANONICAL)

> **재측정 (`final-v1-promotion` step 1, 2026-05-19 v2)** — N_trials anchor를
> 402 (모든 pre-fix trial 누적) → **10** (post-leak-fix model class only) 로 재정의 후
> baseline_v5 pkl 에서 재측정. 정당화는 `experiment_inventory.json.n_trials_active_rationale`.

| 지표 | 값 | 판정 |
|---|---|---|
| Observed SR (annualized) | 1.289 | — |
| **DSR** | **1.470 (p=0.0708)** | **WARN** (p<0.05 strict FAIL이지만 보수적 경계 — legacy 0.43 대비 6배 개선) |
| MinTRL | 1.6년 필요 vs 7.7년 보유 | SUFFICIENT |
| **Grid Haircut** | 0.765 (annualized) | — |
| **Adjusted SR** | **0.524** | **PASS** (legacy 0.062 대비 8.5배 개선) |
| Late entrants | 0 | CLEAN |
| Sub-period IR | P1 +0.72 / P2 +1.26 / P3 +1.86 | STABLE (모두 양수) |
| **Overall verdict** | **WARN** | DSR borderline; Haircut clear PASS |

**해석**: corrected N=10 하에서 baseline_v5 의 adjusted SR 0.524 는 legacy 0.062 (단위 오류 + N=402 동시 inflation) 의 8.5 배. DSR p=0.0708 은 α=0.05 strict 기준 FAIL 이지만 weak ACCEPT 로 해석되는 borderline. Haircut 은 clear PASS. **Promotion-eligible 로 판정**.

**재산출 명령:**
```bash
python run_selection_bias.py --auto --label baseline_v5
# Report: outputs/baseline_v5/selection_bias_report.md
# CSV:    outputs/csv/selection_bias_metrics.csv
```

---

### Selection Bias 검증 (legacy environment — STALE 2026-05-19, audit 보존용)

> **⚠️ 아래 측정은 data-leakage-fix 이전(누수 환경)의 production strategy(`iter15_65tkr_reb21_vtg`)에서**
> **이뤄진 것**이다. 위의 새 박스(`baseline_v5` recount, 2026-05-19 v2)가 canonical. 본 항목은
> 단위 오류 이력 + N=402 inflation 이력을 audit trail 로 보존하기 위해 유지.

원본 측정값 (`run_selection_bias.py --auto --label iter15_65tkr_reb21_vtg`, 2026-04-30, 누수 환경):

> **이전 자료 (`DSR=25.96`, `Haircut=0.078`, `Adjusted SR=0.585`, verdict=PASS) 는 단위 오류 결과이므로 폐기.**
> Bailey-LdP DSR/Haircut 계산에서 `observed_SR`은 annualized인데 `sigma_SR`/`E_max_SR`/haircut은 daily scale로 섞여 있어 DSR이 약 √252 ≈ 15.9배 부풀려졌고 haircut은 같은 배수만큼 축소돼 있었음.
> 본 항목은 단위 통일 후 (`run_selection_bias.py` 70-88, 133-155) 65종목 / IR 1.31 production pkl에서 다시 측정한 값.

| 지표 | 값 | 판정 |
|---|---|---|
| Observed SR (annualized) | 1.299 | — |
| **DSR** | **0.174 (p=0.4309)** | **FAIL** (p ≥ 0.05; SR이 N=402 다중비교 null의 expected max를 의미있게 넘지 못함) |
| MinTRL | 1.6년 필요 vs 7.7년 보유 | SUFFICIENT |
| **Grid Haircut** | **1.237** (annualized) | — |
| **Adjusted SR** | **0.062** | PASS (margin 매우 작음) |
| Late entrants | 0 | CLEAN |
| Sub-period IR | P1 1.53 / P2 0.23 / P3 1.91 | STABLE (모두 양수) |
| **Overall verdict** | **FAIL** | DSR p-value 미달 |

**해석:** Haircut은 약식 (positive after haircut만 요구) 통과지만 DSR은 강한 검정에서 미달. 즉 production strategy의 `IR=1.31`은 N_trials=402 다중비교 보정 하에 통계적으로 유의하지 않을 수 있음. 다음 중 하나가 필요:
1. N_trials 정의 재검토 (실제 distinct 후보 수가 402보다 작을 가능성 — 동일 axis의 grid 중복 제외)
2. 더 긴 OOS 보유로 SR 안정성 입증
3. 신호/포트폴리오 개선으로 SR 상향 (현재 1.30 → 목표 ~1.65 = 1.96-σ)

**재산출 명령 (legacy 환경 audit 재현용):**
```bash
python run_selection_bias.py --auto --label iter15_65tkr_reb21_vtg
# 또는 explicit pkl path (현재 archive 디렉터리):
python run_selection_bias.py --auto --pkl outputs/baseline_v4_legacy/backtest_result.pkl
```
fallback chain: top-level `outputs/backtest_result.pkl` → `baseline_v4/` (= current deploy alias).

### 검증 체크리스트 (research gate denominator = iter15_FINAL_postfix, production deploy = baseline_v5_deploy, 2026-05-19 v2)
1. ✅ **Look-ahead bias 없음** — backtest.py 실행 타이밍 + walk-forward embargo (data-leakage-fix Task A)
2. ⚠️ **Selection bias gate** — baseline_v5 recount (N=10) DSR p=0.0708 WARN / Adjusted SR 0.524 PASS / Haircut clear PASS. legacy 환경 DSR FAIL 결과는 STALE (위 STALE 박스 참조)
3. ✅ **Score↔Position 일치** — 모든 OW는 z>0 종목 (score-gate 강제, `enforce_score_gated_ow=True`)
4. ✅ **Core-Satellite 구조** — satellite_budget 0.225 → ~78% 코어 + ~22% 새틀라이트
5. ✅ **TE 안정** — 2.88% (research) / 3.12% (production deploy)
6. ✅ **Raw revision 노이즈 제거** — `clean_revision_spikes(mode="reversion_gated")`. Task B ablation 결과 down_only 대안 CI [-0.161, +0.737] (parsimony DROP) — 현재 default 유지
7. ✅ **금융주 유니버스 복원** — JPM/GS essential-sheet 교집합 (REDESIGN I)
8. ✅ **Turnover 제어** — research 90.8% two-way / production deploy 102.4% (gate ceiling 118.6%)
9. ✅ **Long-Only IR ≥ 1.0 (production)** — production deploy IR=1.408 ✅ / research baseline IR=0.392 (gate denominator 용도; promotion 비교만)
10. ⚠️ **Sub-period IR — diagnostic only (2026-05-19)** — production P1=+0.67 / P2=+1.52 / P3=+1.74. Sub-period IRs are NO LONGER promotion gates per Task C step 3; primary gate is rolling IR + SPA p-value (docs/BASELINE.md). Sub-period 수치는 regime inspection용으로만 유지
11. ⚠️ **Value-trap gate** — `value_trap_gate_enabled=True` 유지하되 `vtg_scale=0.0`이라 실효 비활성. Task B ablation 결과 ΔIR=-0.046 CI=[-0.316, +0.186] (no gate evidence) → scale 0 으로 풀어두고 enable flag 만 호환성 위해 켜둠
12. ✅ **Walk-forward embargo** — `embargo_days=20` (= forward_horizon). train_end ~ val_start, val_end ~ predict 사이 20일 갭. data-leakage-fix Task A step 0
13. ✅ **OOS hold-out 자동 강제** — research/oos_verify/deploy/production(deprecated) 모드. cutoff=2024-12-31. peek 카운터 `experiment_inventory.json.n_oos_peeks`로 회계. Task A step 1
14. ✅ **Single-statistic primary gate (rolling IR + SPA)** — `src/analytics.rolling_ir_stats`, `spa_pvalue` (Hansen 2005 simplified). `compute_metrics`에 7개 신규 키. 다중-목표 fitting 방지. selection-bias-discipline Task C step 2
15. ✅ **Checkpoint config fingerprint** — `src/backtest.compute_config_fingerprint`. Phase 1/2/4 캐시가 silent stale로 재사용되지 않도록 mismatch 시 자동 폐기. Task C step 0
16. ✅ **Turnover gate effective ceiling** — 1.1864 (= max(research baseline 0.908, legacy deploy 1.1364) + 5pp). 이전 "≥ baseline + 5pp"는 embargo-trimmed research turnover (0.908)만 기준 삼아 ceiling이 인위적으로 낮았음 (0.958). 실제 production이 지불하는 turnover는 deploy baseline (1.1364)에 의해 결정되므로 ceiling은 둘 중 큰 값. `docs/BASELINE.md` 2026-05-19 v2 gate (5) 참조. baseline_v5 (1.1031) PROMOTION-ELIGIBLE.
17. ✅ **Rolling IR min gate relativized** — `rolling_ir_min` 절대 임계값 `≥ -0.20`은 research baseline 자신 (-2.15)도 통과 못 했던 logic error. `≥ baseline - 0.20`으로 수정 (gate 1·2와 일관된 relative 형식). `docs/BASELINE.md` 2026-05-19 v2 gate (3) 참조.
