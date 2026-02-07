from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration untuk VECM project."""

    # Data Download Settings
    vecm_price_download: str = "auto"
    vecm_input: Path = Path("vecm_project/data/adj_close_data.csv")
    vecm_yf_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )
    vecm_yf_impersonate: str = "chrome124"
    vecm_yf_verify: Optional[str] = None
    vecm_proxy_auth: Optional[str] = None
    offline_fallback_path: Optional[Path] = None

    # Execution Settings
    vecm_max_workers: int = 4
    vecm_stage: Optional[str] = None
    vecm_prefilter: str = "off"

    # Playbook Settings
    playbook_input_file: Optional[Path] = None
    playbook_subset: Optional[str] = None
    playbook_method: Optional[str] = None
    playbook_roll_years: Optional[float] = None
    playbook_oos_start: Optional[str] = None
    playbook_horizon: Optional[str] = None
    playbook_stage: Optional[int] = None
    playbook_notes: Optional[str] = None
    playbook_exit: Optional[str] = None
    playbook_z_entry: Optional[float] = None
    playbook_z_exit: Optional[float] = None
    playbook_z_stop: Optional[float] = None
    playbook_max_hold: Optional[int] = None
    playbook_cooldown: Optional[int] = None
    playbook_z_auto_method: Optional[str] = None
    playbook_z_auto_q: Optional[float] = None
    playbook_z_entry_cap: Optional[float] = None
    playbook_gate_require_corr: Optional[int] = None
    playbook_gate_corr_min: Optional[float] = None
    playbook_gate_corr_win: Optional[int] = None
    playbook_gate_enforce: Optional[bool] = None
    playbook_short_filter: Optional[bool] = None
    playbook_beta_weight: Optional[bool] = None
    playbook_cost_bps: Optional[float] = None
    playbook_half_life_max: Optional[float] = None
    playbook_dd_stop: Optional[float] = None
    playbook_fee_buy: Optional[float] = None
    playbook_fee_sell: Optional[float] = None
    playbook_p_th: Optional[float] = None
    playbook_regime_confirm: Optional[int] = None
    playbook_long_only: Optional[bool] = None
    playbook_kelly_frac: Optional[float] = None
    playbook_vol_cap: Optional[float] = None
    playbook_ann_days: Optional[int] = None
    playbook_debug: Optional[bool] = None
    playbook_selftest: Optional[bool] = None
    playbook_seed: Optional[int] = None
    playbook_tag: Optional[str] = None
    playbook_mom_enable: Optional[bool] = None
    playbook_mom_z: Optional[float] = None
    playbook_mom_k: Optional[int] = None
    playbook_mom_gate_k: Optional[int] = None
    playbook_mom_cooldown: Optional[int] = None
    playbook_outlier_iqr_mult: Optional[float] = None
    playbook_outlier_max_ratio: Optional[float] = None

    # Notification Settings (Optional)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_pass: Optional[str] = None
    smtp_to: Optional[str] = None
    smtp_from: Optional[str] = None
    smtp_starttls: bool = True

    @field_validator(
        "playbook_z_exit",
        "playbook_z_stop",
        "playbook_z_auto_q",
        "playbook_z_entry_cap",
        "playbook_gate_corr_min",
        "playbook_dd_stop",
        "playbook_fee_buy",
        "playbook_fee_sell",
        "playbook_p_th",
        "playbook_kelly_frac",
        "playbook_vol_cap",
        "playbook_outlier_max_ratio",
    )
    @classmethod
    def _validate_unit_interval(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not 0 <= value <= 1:
            raise ValueError("value must be between 0 and 1")
        return value

    @field_validator(
        "playbook_roll_years",
        "playbook_cost_bps",
        "playbook_half_life_max",
        "playbook_outlier_iqr_mult",
    )
    @classmethod
    def _validate_positive_float(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator(
        "playbook_max_hold",
        "playbook_cooldown",
        "playbook_gate_corr_win",
        "playbook_regime_confirm",
        "playbook_ann_days",
        "playbook_mom_k",
        "playbook_mom_gate_k",
        "playbook_mom_cooldown",
    )
    @classmethod
    def _validate_positive_int(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("playbook_stage")
    @classmethod
    def _validate_stage(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value < 0:
            raise ValueError("stage must be non-negative")
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
settings = Settings()
