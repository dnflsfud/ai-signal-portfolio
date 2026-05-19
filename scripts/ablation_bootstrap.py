"""Block-bootstrap 95% CI for ΔIR between two active-return series.

Used by Task B step 2 to attach statistical confidence to ablation deltas.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so this works when imported from notebooks.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _annualised_ir(active: np.ndarray) -> float:
    a = active[~np.isnan(active)]
    if len(a) < 20 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * np.sqrt(252))


def block_bootstrap_delta_ir(
    base_active: pd.Series,
    var_active: pd.Series,
    block_size: int = 10,
    n_iter: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Stationary-block bootstrap for ΔIR = IR(var) − IR(base).

    The two series are aligned on their common date index; only the
    intersection is bootstrapped (any dates missing from either side are
    dropped). Resampling uses a moving-block scheme: for each iteration,
    draw ceil(n/block_size) blocks of consecutive length block_size from a
    common index pool and apply the SAME index sequence to both series so
    paired structure is preserved (otherwise the diff variance would be
    inflated by uncorrelated noise).
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")

    common = base_active.index.intersection(var_active.index)
    if len(common) < 50:
        return {
            "n_observations": int(len(common)),
            "delta_ir_observed": float("nan"),
            "delta_ir_mean": float("nan"),
            "delta_ir_lo95": float("nan"),
            "delta_ir_hi95": float("nan"),
            "p_value_two_sided": float("nan"),
        }

    a_base = base_active.reindex(common).to_numpy(dtype=float)
    a_var = var_active.reindex(common).to_numpy(dtype=float)
    n = len(common)

    observed = _annualised_ir(a_var) - _annualised_ir(a_base)

    rng = np.random.default_rng(seed)
    # Pre-compute number of blocks needed
    n_blocks = (n + block_size - 1) // block_size
    deltas = np.empty(n_iter, dtype=float)
    max_start = n - block_size + 1
    if max_start < 1:
        max_start = 1

    for i in range(n_iter):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        idx = idx[idx < n][:n]  # trim to original length
        deltas[i] = _annualised_ir(a_var[idx]) - _annualised_ir(a_base[idx])

    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    # Two-sided p-value under H0: ΔIR = 0, centring the bootstrap distribution.
    centred = deltas - np.nanmean(deltas)
    p = float(np.mean(np.abs(centred) >= abs(observed)))
    return {
        "n_observations": int(n),
        "delta_ir_observed": float(observed),
        "delta_ir_mean": float(np.nanmean(deltas)),
        "delta_ir_lo95": float(lo),
        "delta_ir_hi95": float(hi),
        "p_value_two_sided": p,
    }


def _smoke():
    """Hand-rolled smoke test."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    a = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    b = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    out = block_bootstrap_delta_ir(a, b, n_iter=200)
    print("smoke:", out)
    expected = {"delta_ir_observed", "delta_ir_mean",
                "delta_ir_lo95", "delta_ir_hi95", "p_value_two_sided"}
    assert expected <= set(out.keys()), out
    # Strong-alpha case: variant beats baseline by 30 bp/day on 1% vol
    # → mean t ≈ 30 → CI strictly above zero even at modest n_iter.
    c = b + 0.003
    out2 = block_bootstrap_delta_ir(a, c, n_iter=500)
    print("strong-alpha:", out2)
    assert out2["delta_ir_observed"] > 0, out2
    assert out2["delta_ir_lo95"] > 0, "CI lower bound should clear zero with strong alpha"
    print("OK")


if __name__ == "__main__":
    _smoke()
