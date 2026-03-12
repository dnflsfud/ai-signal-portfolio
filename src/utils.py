"""유틸리티 함수."""

import pandas as pd
import numpy as np
from pathlib import Path


def ensure_dir(path: str) -> Path:
    """디렉토리 생성 (존재하지 않으면)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rolling_ic(predictions: pd.DataFrame, realized: pd.DataFrame, window: int = 252) -> pd.Series:
    """Rolling Information Coefficient."""
    ic_daily = pd.Series(index=predictions.index, dtype=float)

    for date in predictions.index:
        if date not in realized.index:
            continue
        pred = predictions.loc[date].dropna()
        real = realized.loc[date].dropna()
        common = pred.index.intersection(real.index)
        if len(common) >= 3:
            ic_daily[date] = pred[common].corr(real[common], method="spearman")

    return ic_daily.rolling(window, min_periods=window // 2).mean()
