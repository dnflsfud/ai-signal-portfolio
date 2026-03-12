"""
Phase 1: 데이터 로드 및 전처리
- RL_Universe_Data.xlsx의 모든 시트를 로드
- 날짜 인덱스 통일 (BusinessDays)
- Sent_Trend 시트 회사명 -> 티커 매핑
- 결측치 처리: ffill -> cross-sectional median
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

# Sent_Trend 시트의 회사명 -> 티커 매핑
COMPANY_TO_TICKER = {
    # 기존
    "Apple": "AAPL", "Microsoft": "MSFT", "Alphabet": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "Nvidia": "NVDA",
    "Tesla": "TSLA", "Palantir": "PLTR", "Broadcom": "AVGO",
    "Micron": "MU", "GE Vernova": "GEV", "Vertiv": "VRT",
    "Bloom Energy": "BE", "Lumentum": "LITE",
    "SK Hynix": "000660", "Samsung Electronics": "005930",
    # Healthcare
    "UnitedHealth": "UNH", "Eli Lilly": "LLY",
    "Intuitive Surgical": "ISRG", "AbbVie": "ABBV", "Regeneron": "REGN",
    # Financials
    "JPMorgan": "JPM", "Visa": "V", "Mastercard": "MA",
    "BlackRock": "BLK", "S&P Global": "SPGI", "Goldman Sachs": "GS",
    # Consumer
    "Costco": "COST", "Home Depot": "HD", "Procter & Gamble": "PG",
    "McDonald's": "MCD", "Walmart": "WMT",
    # Industrials
    "Caterpillar": "CAT", "Honeywell": "HON", "Deere": "DE",
    "Union Pacific": "UNP", "Lockheed Martin": "LMT", "Eaton": "ETN",
    # Energy/Materials/Utilities
    "Exxon Mobil": "XOM", "Cheniere Energy": "LNG",
    "Freeport-McMoRan": "FCX", "Linde": "LIN", "NextEra Energy": "NEE",
    # Real Estate/Infra/Telecom
    "American Tower": "AMT", "Equinix": "EQIX",
    "T-Mobile": "TMUS", "Prologis": "PLD",
    # Tech Diversifier
    "AMD": "AMD", "Salesforce": "CRM", "Netflix": "NFLX",
}

# 시트명 -> 피처 카테고리 매핑
SHEET_CATEGORY = {
    "PX_LAST": "Price",
    "Daily_Returns": "Price",
    "BEST_EPS": "Accounting",
    "BEST_SALES": "Accounting",
    "BEST_PE_RATIO": "Valuation",
    "BEST_PEG_RATIO": "Valuation",
    "BEST_CALCULATED_FCF": "Accounting",
    "BEST_GROSS_MARGIN": "Accounting",
    "CUR_MKT_CAP": "Conditioning",
    "OPER_MARGIN": "Accounting",
    "BEST_CAPEX": "Accounting",
    "BEST_ROE": "Accounting",
    "BEST_PX_BPS_RATIO": "Valuation",
    "BEST_EV_TO_BEST_EBITDA": "Valuation",
    "NEWS_SENTIMENT_DAILY_AVG": "Sentiment",
    "EQY_REC_CONS": "Sellside",
    "Sent_Trend_Momentum_Timeseries": "Sentiment",
    "Sent_Trend_21d_Timeseries": "Sentiment",
    "Factset_EPS_Revision": "Sellside",
    "Factset_Sales_Revision": "Sellside",
    "Factset_TG_Price": "Sellside",
    "Universe_Meta": "Meta",
}

TICKERS = [
    # 기존 16개
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "PLTR", "AVGO", "MU",
    "GEV", "VRT", "BE", "LITE", "000660", "005930",
    # Healthcare
    "UNH", "LLY", "ISRG", "ABBV", "REGN",
    # Financials
    "JPM", "V", "MA", "BLK", "SPGI", "GS",
    # Consumer
    "COST", "HD", "PG", "MCD", "WMT",
    # Industrials / Defense
    "CAT", "HON", "DE", "UNP", "LMT", "ETN",
    # Energy / Materials / Utilities
    "XOM", "LNG", "FCX", "LIN", "NEE",
    # Real Estate / Infra / Telecom
    "AMT", "EQIX", "TMUS", "PLD",
    # Growth / Platform / Tech Diversifier
    "AMD", "CRM", "NFLX",
]

SENT_TREND_SHEETS = {
    "Sent_Trend_Momentum_Timeseries",
    "Sent_Trend_21d_Timeseries",
}

# 날짜 인덱스가 아닌 메타/요약 시트 (전처리에서 제외)
SKIP_SHEETS = {"Universe_Meta", "Summary_Stats", "BusinessDays", "Factor_Meta"}

# Factor 시트 (ticker 기반이 아닌 별도 컬럼 구조)
FACTOR_SHEETS = {"Factor_PX_LAST", "Factor_Returns", "Factor_Meta"}

FACTOR_CATEGORIES = {
    "Market_Index": ["SPX", "NDX", "RTY", "MXWD", "MXEF", "SX5E", "NKY", "HSI", "SHCOMP"],
    "Volatility": ["VIX", "SKEW"],
    "Rates": ["UST_3M", "UST_2Y", "UST_10Y", "US_BEI10", "GER_10Y"],
    "FX": ["DXY", "USDKRW", "USDJPY", "EURUSD", "USDCNH"],
    "Commodity": ["WTI", "GOLD", "COPPER", "BCOM"],
    "Factor_ETF": ["F_MinVol", "F_Quality", "F_HiDiv", "F_Growth", "F_Value", "F_SmCap", "F_HiBeta"],
    "GS_Thematic": ["GS_AI", "GS_Nuclear", "GS_SemiHW"],
    "Macro_Sentiment": ["CESI_US", "AAII_Bull", "AAII_Bear"],
}
ALL_FACTOR_COLUMNS = [col for cols in FACTOR_CATEGORIES.values() for col in cols]


def load_all_sheets(data_path: str) -> Dict[str, pd.DataFrame]:
    """엑셀 파일의 모든 시트를 Dict[시트명, DataFrame]으로 로드."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    xls = pd.ExcelFile(path, engine="openpyxl")
    raw: Dict[str, pd.DataFrame] = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
        raw[sheet_name] = df

    return raw


def _rename_sent_trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sent_Trend 시트의 회사명 컬럼을 티커로 변환."""
    rename_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        for company, ticker in COMPANY_TO_TICKER.items():
            if company.lower() in col_str.lower():
                rename_map[col] = ticker
                break
    return df.rename(columns=rename_map)


def _standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    """인덱스를 DatetimeIndex로 변환."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 문자열로 통일하고, 공백 제거."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 처리: ffill -> 남은 NaN은 해당 날짜 cross-sectional median."""
    df = df.ffill()
    row_medians = df.median(axis=1)
    for col in df.columns:
        mask = df[col].isna()
        if mask.any():
            df.loc[mask, col] = row_medians[mask]
    return df


def _filter_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """TICKERS에 포함된 컬럼만 남김. 컬럼 순서 통일."""
    available = [t for t in TICKERS if t in df.columns]
    return df[available]


def load_universe_meta(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Universe_Meta 시트에서 종목별 섹터 정보 추출."""
    if "Universe_Meta" not in raw:
        sectors = pd.Series("Unknown", index=TICKERS, name="sector")
        return sectors.to_frame()

    meta = raw["Universe_Meta"].copy()
    meta = _standardize_columns(meta)

    # 인덱스가 'AAPL US Equity' 형태이면 티커만 추출
    new_idx = []
    for idx in meta.index:
        idx_str = str(idx).strip()
        # "AAPL US Equity" -> "AAPL", "000660 KS Equity" -> "000660"
        parts = idx_str.split()
        new_idx.append(parts[0] if parts else idx_str)
    meta.index = new_idx

    # Sector 컬럼 표준화
    if "Sector" in meta.columns:
        meta = meta.rename(columns={"Sector": "sector"})

    return meta


def preprocess_sheets(
    raw: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    모든 시트를 전처리하여 반환.
    - DatetimeIndex 통일
    - Sent_Trend 컬럼 매핑
    - 티커 필터링
    - 결측치 처리
    """
    processed: Dict[str, pd.DataFrame] = {}

    for sheet_name, df in raw.items():
        if sheet_name in SKIP_SHEETS or sheet_name in FACTOR_SHEETS:
            continue

        df = df.copy()
        df = _standardize_columns(df)

        # Sent_Trend 시트는 회사명 -> 티커 매핑
        if sheet_name in SENT_TREND_SHEETS:
            df = _rename_sent_trend_columns(df)

        df = _standardize_index(df)
        df = _filter_tickers(df)

        # 수치형 변환
        df = df.apply(pd.to_numeric, errors="coerce")

        # 결측치 처리
        df = _fill_missing(df)

        processed[sheet_name] = df

    return processed


def align_dates(
    processed: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """모든 시트의 날짜 인덱스를 공통 영업일 기준으로 정렬."""
    # 모든 시트의 인덱스 교집합
    common_idx = None
    for df in processed.values():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("시트 간 공통 날짜가 없습니다.")

    common_idx = common_idx.sort_values()

    aligned = {}
    for name, df in processed.items():
        aligned[name] = df.loc[common_idx].copy()

    return aligned


def load_factor_sheets(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Factor_PX_LAST / Factor_Returns를 별도 파이프라인으로 로드."""
    factor_data: Dict[str, pd.DataFrame] = {}
    for sheet_name in ["Factor_PX_LAST", "Factor_Returns"]:
        if sheet_name not in raw:
            continue
        df = raw[sheet_name].copy()
        df = _standardize_columns(df)
        df = _standardize_index(df)
        available = [c for c in ALL_FACTOR_COLUMNS if c in df.columns]
        df = df[available]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.ffill().fillna(0)
        factor_data[sheet_name] = df
    return factor_data


class UniverseData:
    """전처리된 유니버스 데이터를 담는 컨테이너."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw = load_all_sheets(data_path)
        self.meta = load_universe_meta(self.raw)
        self.sheets = preprocess_sheets(self.raw)
        self.sheets = align_dates(self.sheets)
        self.tickers = TICKERS
        self.dates = self.sheets[next(iter(self.sheets))].index

        # Factor 데이터 (별도 파이프라인)
        self.factor_data = load_factor_sheets(self.raw)
        if self.factor_data:
            for name, df in self.factor_data.items():
                common = df.index.intersection(self.dates)
                self.factor_data[name] = df.loc[common]

    @property
    def prices(self) -> pd.DataFrame:
        return self.sheets["PX_LAST"]

    @property
    def returns(self) -> pd.DataFrame:
        return self.sheets["Daily_Returns"]

    @property
    def market_cap(self) -> pd.DataFrame:
        return self.sheets["CUR_MKT_CAP"]

    @property
    def factor_prices(self) -> Optional[pd.DataFrame]:
        return self.factor_data.get("Factor_PX_LAST")

    @property
    def factor_returns(self) -> Optional[pd.DataFrame]:
        return self.factor_data.get("Factor_Returns")

    def has_factor_data(self) -> bool:
        return bool(self.factor_data) and "Factor_Returns" in self.factor_data

    def get_sheet(self, name: str) -> pd.DataFrame:
        if name not in self.sheets:
            raise KeyError(f"시트 '{name}'을 찾을 수 없습니다. 사용 가능: {list(self.sheets.keys())}")
        return self.sheets[name]

    def summary(self) -> str:
        lines = [
            f"데이터 경로: {self.data_path}",
            f"기간: {self.dates[0].strftime('%Y-%m-%d')} ~ {self.dates[-1].strftime('%Y-%m-%d')}",
            f"영업일 수: {len(self.dates)}",
            f"종목 수: {len(self.tickers)}",
            f"시트 수: {len(self.sheets)}",
            "",
            "시트별 shape:",
        ]
        for name, df in self.sheets.items():
            missing_pct = df.isna().sum().sum() / df.size * 100
            lines.append(f"  {name:40s} {str(df.shape):>15s}  결측: {missing_pct:.2f}%")
        if self.factor_data:
            lines.append("")
            lines.append("Factor 데이터:")
            for name, df in self.factor_data.items():
                lines.append(f"  {name:40s} {str(df.shape):>15s}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"UniverseData(tickers={len(self.tickers)}, "
            f"dates={len(self.dates)}, sheets={len(self.sheets)})"
        )
