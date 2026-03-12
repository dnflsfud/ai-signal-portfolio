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
ai_signal_cc/
├── CLAUDE.md
├── data/
│   └── RL_Universe_Data.xlsx
├── src/
│   ├── data_loader.py
│   ├── feature_engine.py
│   ├── target_engine.py
│   ├── model_trainer.py
│   ├── portfolio_optimizer.py
│   ├── attribution.py
│   ├── backtest.py
│   └── utils.py
├── outputs/
└── main.py

## 데이터 사양

### 원본 파일: data/RL_Universe_Data.xlsx
- 기간: 2014.01 ~ 2026.02 (약 4,440 영업일)
- 종목: 15개 (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, PLTR, AVGO, MU, GEV, VRT, BE, LITE, 000660/SK Hynix)

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

최종 피처 수: 약 80~120개.

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

LightGBM params:
  learning_rate=0.05, num_leaves=31, max_depth=6,
  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
  reg_alpha=0.1, reg_lambda=1.0, n_estimators=500

### Phase 5: Walk-Forward 백테스트 (src/backtest.py)
- train_window: 756일(3년)
- retrain_freq: 63일(3개월)
- prediction_horizon: 20일
- rebalance_freq: 5일(주간)

### Phase 6: 포트폴리오 최적화 (src/portfolio_optimizer.py)
cvxpy Mean-Variance Optimization.

목적함수: Maximize(E[r] @ w - lambda * risk - tc * turnover)
- risk_aversion=1.0, turnover_penalty=0.005
- risk = quad_form(w - bm_weights, cov_matrix)
- turnover = norm1(w - prev_weights)

제약: sum(w)=1, w>=0, w<=0.15, 섹터 +-10% vs EW benchmark
Cov matrix: 126일 Ledoit-Wolf shrinkage
벤치마크: 15종목 동일가중(1/15)

### Phase 7: Attribution (src/attribution.py)
- SHAP TreeExplainer
- Feature group별 기여도 (Accounting, Price, Sellside, Conditioning)
- 선형 vs 비선형 분해 (목표 ~50/50)

### Phase 8: 시각화 (main.py)
outputs/ 에 저장:
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
pip install pandas numpy scipy scikit-learn lightgbm cvxpy shap matplotlib plotly openpyxl
python main.py --data_path ./data/RL_Universe_Data.xlsx --output_dir ./outputs/

## 핵심 파라미터
| 파라미터 | 값 | 근거 |
|----------|-----|------|
| 예측 타겟 | 20일 Specific Return | Pictet 기준 |
| PCA 성분 수 | 5 | 15종목 기준 |
| 훈련 기간 | 3년(756일) | 데이터 제약 |
| 재훈련 주기 | 3개월(63일) | Pictet 기준 |
| 리밸런싱 | 주간(5일) | Pictet 기준 |
| Turnover Penalty | 0.005 | 3~4배 감속 |
| 종목 상한 | 15% | 집중 방지 |
| 섹터 편차 | +-10% | 공통 차원 헤지 |

## 검증 체크리스트
1. Look-ahead bias 없음
2. 재훈련 전후 상관 ~0.95
3. Feature importance: Price ~40%, Accounting ~20%, Sellside ~25%, Conditioning ~7-8%
4. 선형/비선형 ~50/50
5. IC > 0.03
6. Long-Only IR >= 1.0
7. 연간 Turnover 150~200%
