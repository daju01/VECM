"""Shared dataclasses for the VECM playbook pipeline."""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class PlaybookConfig:
    input_file: str
    subset: str = ""
    method: str = "TVECM"
    roll_years: float = 3.0
    oos_start: str = ""
    horizon: str = ""
    stage: int = 0
    notes: str = ""
    exit: str = "zexit"
    z_entry: Optional[float] = None
    z_exit: float = 0.55
    z_stop: float = 0.8
    max_hold: int = 8
    min_hold: int = 0
    dynamic_hold: bool = False
    dynamic_hold_max_add: int = 0
    dynamic_hold_step: float = 0.5
    cooldown: int = 1
    z_auto_method: str = "mfpt"
    z_auto_q: float = 0.7
    z_entry_cap: float = 0.85
    gate_require_corr: int = 0
    gate_corr_min: float = 0.60
    gate_corr_win: int = 45
    gate_enforce: bool = True
    short_filter: bool = False
    signal_mode: str = "normal"
    beta_weight: bool = True
    cost_bps: float = 5.0
    half_life_max: float = 120.0
    dd_stop: float = 0.25
    fee_buy: float = 0.0019
    fee_sell: float = 0.0029
    p_th: float = 0.50
    regime_confirm: int = 1
    long_only: bool = True
    kelly_frac: float = 0.5
    vol_cap: float = 0.20
    ann_days: int = 252
    debug: bool = False
    selftest: bool = False
    seed: Optional[int] = None
    tag: str = ""
    mom_enable: bool = False
    mom_z: float = 0.60
    mom_k: int = 2
    mom_gate_k: int = 3
    mom_cooldown: int = 2
    outlier_iqr_mult: float = 3.0
    outlier_max_ratio: float = 0.02

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FeatureConfig:
    base_config: PlaybookConfig
    pair: str
    method: str = "TVECM"
    horizon: str = ""
    data_frame: Optional[pd.DataFrame] = None
    run_id: Optional[str] = None
    macro_exog: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class DecisionParams:
    z_entry: Optional[float]
    z_exit: float
    max_hold: int
    cooldown: int
    p_th: Optional[float] = None
    run_id: Optional[str] = None


@dataclass(frozen=True)
class FeatureBundle:
    run_id: str
    cfg: PlaybookConfig
    pair: str
    selected_l: str
    selected_r: str
    lp: pd.DataFrame
    beta_series: pd.Series
    zect: pd.Series
    combined_gate: pd.Series
    p_mr_series: pd.Series
    delta_score: Optional[pd.Series]
    delta_mom12: Optional[pd.Series]
    alpha_ec: float
    half_life_full: float
    ms_status: str
    ms_error: str
    oos_start_date: dt.date
    horizon: Dict[str, object]


@dataclass(frozen=True)
class FeatureBuildResult:
    features: Optional[FeatureBundle]
    skip_result: Optional[Dict[str, object]] = None
