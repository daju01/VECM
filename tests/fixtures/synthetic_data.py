from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_price_frame(days: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)

    base = 100 + np.cumsum(rng.normal(0.0, 0.4, size=days))
    coin_a = base + rng.normal(0.0, 0.2, size=days)
    coin_b = base * 1.02 + rng.normal(0.0, 0.2, size=days)

    trend_a = 50 + np.cumsum(rng.normal(0.15, 0.5, size=days))
    trend_b = 80 + np.cumsum(rng.normal(0.25, 0.6, size=days))

    noise_a = 100 + rng.normal(0.0, 1.2, size=days)
    noise_b = 95 + rng.normal(0.0, 1.2, size=days)

    market = 200 + np.cumsum(rng.normal(0.05, 0.3, size=days))

    data = {
        "date": dates,
        "COIN_A.JK": np.maximum(coin_a, 1.0),
        "COIN_B.JK": np.maximum(coin_b, 1.0),
        "TREND_A.JK": np.maximum(trend_a, 1.0),
        "TREND_B.JK": np.maximum(trend_b, 1.0),
        "NOISE_A.JK": np.maximum(noise_a, 1.0),
        "NOISE_B.JK": np.maximum(noise_b, 1.0),
        "^JKSE": np.maximum(market, 1.0),
    }
    return pd.DataFrame(data)


def write_synthetic_csv(path: Path, days: int = 260, seed: int = 7) -> pd.DataFrame:
    frame = make_synthetic_price_frame(days=days, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
