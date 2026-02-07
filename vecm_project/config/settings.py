from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration untuk VECM project."""

    # Data Download Settings
    vecm_price_download: str = "auto"
    vecm_input: Path = Path("vecm_project/data/adj_close_data.csv")
    vecm_yf_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )
    vecm_yf_impersonate: Optional[str] = None
    vecm_yf_verify: str = "true"
    vecm_proxy_auth: Optional[str] = None
    offline_fallback_path: Optional[Path] = None

    # Execution Settings
    vecm_max_workers: int = 4
    vecm_stage: Optional[str] = None
    vecm_prefilter: str = "off"

    # Playbook Settings
    playbook_fee_buy: Optional[float] = None
    playbook_fee_sell: Optional[float] = None

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
settings = Settings()
