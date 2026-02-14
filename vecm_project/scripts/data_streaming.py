"""Streaming adjusted close data loader for Indonesian equities universe."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import time
import threading
from functools import lru_cache
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from requests.utils import get_environ_proxies

try:  # pragma: no cover - optional dependency for robust downloads
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - curl_cffi not available
    curl_requests = None

from vecm_project.config.settings import settings

from . import storage
from .cache_keys import hash_dataframe

LOGGER = storage.configure_logging("data_streaming")
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "adj_close_data.csv"
VOLUME_PATH = DATA_DIR / "volume_data.csv"
CACHE_META_PATH = DATA_DIR / "adj_close_data.meta.json"
CACHE_ROOT = BASE_DIR / "cache"
TICKER_CACHE_DIR = CACHE_ROOT / "tickers"
TICKER_CONFIG_PATH = BASE_DIR / "config" / "ticker_groups.json"
TICKER_CONFIG_ENV = "VECM_TICKER_CONFIG"
OFFLINE_FALLBACK_PATH = settings.offline_fallback_path or (DATA_DIR / "offline_prices.csv")
DEFAULT_START_DATE = dt.date(2013, 1, 1)
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.75
_CIRCUIT_BREAKER = {"failures": 0, "opened_until": None}
CB_MAX_FAILURES = 3
CB_COOLDOWN_MINUTES = 30
DOWNLOAD_CONTROL_ENV = "VECM_PRICE_DOWNLOAD"
_DOWNLOAD_DISABLE = {"0", "false", "no", "off", "skip", "never"}
_DOWNLOAD_FORCE = {"force", "always"}
YF_USER_AGENT_ENV = "VECM_YF_USER_AGENT"
YF_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
YF_RETRY_STATUS_CODES = (403, 429, 500, 502, 503, 504)
YF_IMPERSONATE_ENV = "VECM_YF_IMPERSONATE"
YF_VERIFY_ENV = "VECM_YF_VERIFY"
YF_PROXY_AUTH_ENV = "VECM_PROXY_AUTH"


_REQUESTS_SESSION: Any = None
_REQUESTS_SESSION_PID: Optional[int] = None
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_LAST_CALL: Optional[float] = None


def _rate_limit_pause() -> None:
    """Throttle outbound requests to respect external API limits."""
    rate_limit = settings.vecm_rate_limit_per_sec
    if not rate_limit or rate_limit <= 0:
        return
    min_interval = 1.0 / rate_limit
    global _RATE_LIMIT_LAST_CALL
    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        if _RATE_LIMIT_LAST_CALL is None:
            _RATE_LIMIT_LAST_CALL = now
            return
        elapsed = now - _RATE_LIMIT_LAST_CALL
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            now = time.monotonic()
        _RATE_LIMIT_LAST_CALL = now


MONITORING_DIR = BASE_DIR / "out_ms" / "monitoring"


def _write_download_metrics(
    *,
    refreshed: Sequence[str],
    failed: Sequence[str],
    throttled: Sequence[str],
    duration_s: float,
) -> None:
    """Persist a small JSON snapshot for monitoring the download pipeline."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "refreshed": list(refreshed),
        "failed": list(failed),
        "skipped": list(throttled),
        "duration_s": float(duration_s),
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
    path = MONITORING_DIR / "price_download.json"
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - best effort observability
        LOGGER.warning("Failed to write download metrics: %s", exc)


DEFAULT_TICKER_CONFIG: Dict[str, List[str]] = {
    "macro_tickers": ["^JKSE", "USDIDR=X"],
    "commodity_price": ["GC=F"],
    "banking_group": [
        "BBCA.JK",
        "BMRI.JK",
        "BBRI.JK",
        "BBNI.JK",
        "BNLI.JK",
        "MEGA.JK",
        "BNGA.JK",
        "PNBN.JK",
        "BRIS.JK",
        "BDMN.JK",
        "ARTO.JK",
    ],
    "telco_group": ["TLKM.JK", "ISAT.JK", "EXCL.JK", "FREN.JK", "MTEL.JK", "TOWR.JK", "TBIG.JK"],
    "datacenter_group": ["DCII.JK", "EDGE.JK"],
    "digital_platform_group": [
        "BUKA.JK",
        "GOTO.JK",
        "EMTK.JK",
        "MCAS.JK",
        "SCMA.JK",
        "TECH.JK",
        "BELI.JK",
    ],
    "coal_group": [
        "ADRO.JK",
        "PTBA.JK",
        "HRUM.JK",
        "BYAN.JK",
        "BUMI.JK",
        "ITMG.JK",
        "DSSA.JK",
        "CUAN.JK",
        "TOBA.JK",
        "GEMS.JK",
        "SMMT.JK",
        "INDY.JK",
    ],
    "mineral_group": ["TINS.JK", "BRMS.JK", "NICL.JK"],
    "strategic_resource_group": ["MDKA.JK", "AMMN.JK", "ANTM.JK", "INCO.JK", "MBMA.JK", "NCKL.JK"],
    "energy_chemical_group": ["ESSA.JK", "RAJA.JK", "BRPT.JK", "BREN.JK", "DSSA.JK", "PGAS.JK", "INDY.JK"],
    "food_beverage_group": ["ICBP.JK", "INDF.JK", "MYOR.JK", "ULTJ.JK", "CMRY.JK", "CLEO.JK", "ROTI.JK", "JPFA.JK"],
    "non_food_consumer_group": ["UNVR.JK", "HMSP.JK", "GGRM.JK", "KLBF.JK", "SIDO.JK", "WIIM.JK"],
    "industry_infra_group": [
        "ASII.JK",
        "UNTR.JK",
        "SMGR.JK",
        "WTON.JK",
        "PTPP.JK",
        "JSMR.JK",
        "AKRA.JK",
        "KRAS.JK",
        "SMBR.JK",
        "WIKA.JK",
    ],
    "investment_group": ["SRTG.JK", "MIDI.JK", "SCMA.JK", "FILM.JK", "MNCN.JK", "LPPF.JK"],
    "petrochemical_group": ["TPIA.JK", "BRPT.JK"],
    "palmoil_group": ["AALI.JK", "LSIP.JK", "SIMP.JK", "DSNG.JK", "SMAR.JK", "TBLA.JK", "BWPT.JK", "TAPG.JK", "PALM.JK"],
    "multifinance_group": ["ADMF.JK", "BFIN.JK", "MFIN.JK", "IMJS.JK", "TIFA.JK", "CFIN.JK", "TFCM.JK", "BCAP.JK"],
    "media_retail_group": ["SCMA.JK", "MNCN.JK", "FILM.JK", "LPPF.JK"],
    "consumer_retail_group": ["AMRT.JK", "MDIY.JK", "DNET.JK"],
    "property_group": ["PANI.JK", "BSDE.JK", "CTRA.JK", "SMRA.JK", "ASRI.JK", "BKSL.JK", "DMAS.JK", "LPKR.JK", "LCGP.JK"],
    "healthcare_providers": ["MIKA.JK", "PRDA.JK", "SAME.JK", "SILO.JK", "HEAL.JK"],
    "logistics_transport": ["ASSA.JK", "BIRD.JK"],
    "shipping": ["SMDR.JK"],
    "aviation": ["GIAA.JK"],
    "cement_materials": ["INTP.JK"],
    "specialty_retail": ["RALS.JK", "ERAA.JK", "MAPI.JK", "ACES.JK"],
    "poultry_feed": ["MAIN.JK", "CPIN.JK"],
    "banking_additional": ["BTPS.JK", "BTPN.JK"],
    "industrial_property": ["SSIA.JK", "BEST.JK"],
    "pulp_paper": ["INKP.JK", "TKIM.JK"],
    "energy_resources": ["MEDC.JK"],
    "renewables_utilities": ["KEEN.JK", "ARKO.JK", "PGEO.JK"],
}


def _circuit_allows() -> bool:
    opened_until = _CIRCUIT_BREAKER["opened_until"]
    if opened_until is None:
        return True
    if dt.datetime.now(dt.timezone.utc) >= opened_until:
        _CIRCUIT_BREAKER["opened_until"] = None
        _CIRCUIT_BREAKER["failures"] = 0
        return True
    return False


def _circuit_record_failure() -> None:
    _CIRCUIT_BREAKER["failures"] += 1
    if _CIRCUIT_BREAKER["failures"] >= CB_MAX_FAILURES:
        _CIRCUIT_BREAKER["opened_until"] = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
            minutes=CB_COOLDOWN_MINUTES
        )


def _circuit_record_success() -> None:
    _CIRCUIT_BREAKER["failures"] = 0
    _CIRCUIT_BREAKER["opened_until"] = None


def _download_alpha_vantage(ticker: str, start: pd.Timestamp, end: dt.date) -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={ticker}&outputsize=full&apikey={api_key}"
    )
    try:
        _rate_limit_pause()
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - network/runtime errors
        LOGGER.warning("Alpha Vantage fallback failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    series = payload.get("Time Series (Daily)") or {}
    if not series:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    rows = []
    for date_str, values in series.items():
        if not values:
            continue
        rows.append(
            {
                "Date": pd.to_datetime(date_str),
                "Ticker": ticker,
                "AdjClose": float(values.get("5. adjusted close", "nan")),
                "Volume": float(values.get("6. volume", "nan")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame[(frame["Date"] >= start) & (frame["Date"] <= pd.Timestamp(end))]
    return frame


def _download_fallback_provider(ticker: str, start: pd.Timestamp, end: dt.date) -> pd.DataFrame:
    alpha_frame = _download_alpha_vantage(ticker, start, end)
    if not alpha_frame.empty:
        return alpha_frame
    return _offline_prices_for(ticker, start, end)


def _normalise_ticker_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {}
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            cleaned[key] = [str(item) for item in value if str(item).strip()]
    return cleaned


def _copy_ticker_config(config: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {key: list(values) for key, values in config.items()}


def load_ticker_config(config_path: Optional[pathlib.Path] = None) -> Dict[str, List[str]]:
    path = config_path or pathlib.Path(os.getenv(TICKER_CONFIG_ENV, str(TICKER_CONFIG_PATH)))
    if not path.exists():
        LOGGER.warning("Ticker config not found at %s; using defaults", path)
        return _copy_ticker_config(DEFAULT_TICKER_CONFIG)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Failed to read ticker config at %s: %s; using defaults", path, exc)
        return _copy_ticker_config(DEFAULT_TICKER_CONFIG)
    if not isinstance(payload, dict):
        LOGGER.warning("Ticker config at %s must be a JSON object; using defaults", path)
        return _copy_ticker_config(DEFAULT_TICKER_CONFIG)
    cleaned = _normalise_ticker_config(payload)
    if not cleaned:
        LOGGER.warning("Ticker config at %s is empty after normalisation; using defaults", path)
        return _copy_ticker_config(DEFAULT_TICKER_CONFIG)
    merged = _copy_ticker_config(DEFAULT_TICKER_CONFIG)
    merged.update(cleaned)
    return merged


TICKER_CONFIG = load_ticker_config()
MACRO_TICKERS = TICKER_CONFIG.get("macro_tickers", [])
COMMODITY_PRICE = TICKER_CONFIG.get("commodity_price", [])
BANKING_GROUP = TICKER_CONFIG.get("banking_group", [])
TELCO_GROUP = TICKER_CONFIG.get("telco_group", [])
DATACENTER_GROUP = TICKER_CONFIG.get("datacenter_group", [])
DIGITAL_PLATFORM_GROUP = TICKER_CONFIG.get("digital_platform_group", [])
COAL_GROUP = TICKER_CONFIG.get("coal_group", [])
MINERAL_GROUP = TICKER_CONFIG.get("mineral_group", [])
STRATEGIC_RESOURCE_GROUP = TICKER_CONFIG.get("strategic_resource_group", [])
ENERGY_CHEMICAL_GROUP = TICKER_CONFIG.get("energy_chemical_group", [])
FOOD_BEVERAGE_GROUP = TICKER_CONFIG.get("food_beverage_group", [])
NON_FOOD_CONSUMER_GROUP = TICKER_CONFIG.get("non_food_consumer_group", [])
INDUSTRY_INFRA_GROUP = TICKER_CONFIG.get("industry_infra_group", [])
INVESTMENT_GROUP = TICKER_CONFIG.get("investment_group", [])
PETROCHEMICAL_GROUP = TICKER_CONFIG.get("petrochemical_group", [])
PALMOIL_GROUP = TICKER_CONFIG.get("palmoil_group", [])
MULTIFINANCE_GROUP = TICKER_CONFIG.get("multifinance_group", [])
MEDIA_RETAIL_GROUP = TICKER_CONFIG.get("media_retail_group", [])
CONSUMER_RETAIL_GROUP = TICKER_CONFIG.get("consumer_retail_group", [])
PROPERTY_GROUP = TICKER_CONFIG.get("property_group", [])
HEALTHCARE_PROVIDERS = TICKER_CONFIG.get("healthcare_providers", [])
LOGISTICS_TRANSPORT = TICKER_CONFIG.get("logistics_transport", [])
SHIPPING = TICKER_CONFIG.get("shipping", [])
AVIATION = TICKER_CONFIG.get("aviation", [])
CEMENT_MATERIALS = TICKER_CONFIG.get("cement_materials", [])
SPECIALTY_RETAIL = TICKER_CONFIG.get("specialty_retail", [])
POULTRY_FEED = TICKER_CONFIG.get("poultry_feed", [])
BANKING_ADDITIONAL = TICKER_CONFIG.get("banking_additional", [])
INDUSTRIAL_PROPERTY = TICKER_CONFIG.get("industrial_property", [])
PULP_PAPER = TICKER_CONFIG.get("pulp_paper", [])
ENERGY_RESOURCES = TICKER_CONFIG.get("energy_resources", [])
RENEWABLES_UTILITIES = TICKER_CONFIG.get("renewables_utilities", [])

ALL_TICKERS = sorted(set(chain.from_iterable(TICKER_CONFIG.values())))


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TICKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_cache_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def _ticker_cache_dir(ticker: str) -> pathlib.Path:
    return TICKER_CACHE_DIR / _safe_cache_slug(ticker)


def _ticker_cache_path(ticker: str, data_hash: str, suffix: str) -> pathlib.Path:
    return _ticker_cache_dir(ticker) / f"{data_hash}{suffix}"


def _find_existing_ticker_cache(ticker: str, data_hash: Optional[str] = None) -> Optional[pathlib.Path]:
    cache_dir = _ticker_cache_dir(ticker)
    if data_hash:
        for suffix in (".parquet", ".feather"):
            candidate = _ticker_cache_path(ticker, data_hash, suffix)
            if candidate.exists():
                return candidate
        return None
    if not cache_dir.exists():
        return None
    candidates = sorted(cache_dir.glob("*.parquet"))
    if not candidates:
        candidates = sorted(cache_dir.glob("*.feather"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_ticker_cache(path: pathlib.Path, ticker: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_feather(path)
    if "Date" not in frame.columns:
        raise ValueError(f"Ticker cache at {path} missing Date column")
    if "AdjClose" not in frame.columns:
        raise ValueError(f"Ticker cache at {path} missing AdjClose column")
    if "Volume" not in frame.columns:
        frame["Volume"] = float("nan")
    frame = frame[["Date", "AdjClose", "Volume"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Ticker"] = ticker
    frame["AdjClose"] = pd.to_numeric(frame["AdjClose"], errors="coerce")
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    return frame.dropna(subset=["AdjClose"])


def _write_ticker_cache(ticker: str, frame: pd.DataFrame) -> str:
    cache_dir = _ticker_cache_dir(ticker)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if "Volume" not in frame.columns:
        frame = frame.assign(Volume=float("nan"))
    tidy = frame[["Date", "AdjClose", "Volume"]].copy()
    tidy["Date"] = pd.to_datetime(tidy["Date"])
    tidy["AdjClose"] = pd.to_numeric(tidy["AdjClose"], errors="coerce")
    tidy["Volume"] = pd.to_numeric(tidy["Volume"], errors="coerce")
    tidy = tidy.dropna(subset=["AdjClose"]).sort_values("Date")
    data_hash = hash_dataframe(tidy)
    path = _ticker_cache_path(ticker, data_hash, ".parquet")
    try:
        tidy.to_parquet(path, index=False)
        return data_hash
    except Exception:  # pragma: no cover - fallback to feather
        path = _ticker_cache_path(ticker, data_hash, ".feather")
        tidy.to_feather(path)
    return data_hash


def _get_requests_session() -> Any:
    global _REQUESTS_SESSION, _REQUESTS_SESSION_PID
    current_pid = os.getpid()
    if _REQUESTS_SESSION is not None and _REQUESTS_SESSION_PID == current_pid:
        return _REQUESTS_SESSION

    user_agent = settings.vecm_yf_user_agent or YF_DEFAULT_USER_AGENT
    impersonate = settings.vecm_yf_impersonate or "chrome124"
    verify_path = (settings.vecm_yf_verify or "").strip() or os.getenv("CODEX_PROXY_CERT", "").strip()
    if not verify_path:
        for env_var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
            verify_candidate = os.getenv(env_var, "").strip()
            if verify_candidate:
                verify_path = verify_candidate
                break
    proxy_auth = (settings.vecm_proxy_auth or "").strip()
    proxies = get_environ_proxies("https://query1.finance.yahoo.com")

    if curl_requests is not None:
        session = curl_requests.Session(impersonate=impersonate)
        session.headers["User-Agent"] = user_agent
        session.headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        session.headers["Accept-Language"] = "en-US,en;q=0.9"
        session.headers["Host"] = "query1.finance.yahoo.com"
        session.headers["Connection"] = "keep-alive"
        session.headers["Proxy-Connection"] = "keep-alive"
        session.headers["Accept-Encoding"] = "gzip, deflate, br"
        session.timeout = 30
    else:
        session = requests.Session()
        session.headers["User-Agent"] = user_agent
        session.headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        session.headers["Accept-Language"] = "en-US,en;q=0.9"
        session.headers["Host"] = "query1.finance.yahoo.com"
        session.headers["Connection"] = "keep-alive"
        session.headers["Proxy-Connection"] = "keep-alive"
        session.headers["Accept-Encoding"] = "gzip, deflate, br"
        retries = Retry(
            total=5,
            read=5,
            connect=5,
            backoff_factor=0.5,
            status_forcelist=YF_RETRY_STATUS_CODES,
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

    if proxies:
        session.proxies = proxies
        if proxy_auth:
            session.headers["Proxy-Authorization"] = proxy_auth
    else:
        session.proxies = {}

    if verify_path:
        session.verify = verify_path

    _REQUESTS_SESSION = session
    _REQUESTS_SESSION_PID = current_pid
    return session


def _read_existing_prices() -> Optional[pd.DataFrame]:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    if {"Ticker", "AdjClose"}.issubset(df.columns):
        tidy = df[["Date", "Ticker", "AdjClose"]].copy()
        tidy["Volume"] = float("nan")
        tidy["Date"] = pd.to_datetime(tidy["Date"])
        return tidy[["Date", "Ticker", "AdjClose", "Volume"]]
    if "Date" not in df.columns:
        LOGGER.warning("Existing price file missing 'Date' column; ignoring cache at %s", DATA_PATH)
        return None
    value_cols = [col for col in df.columns if col != "Date"]
    if not value_cols:
        LOGGER.warning("Existing price cache has no ticker columns; ignoring cache at %s", DATA_PATH)
        return None
    tidy = df.melt(id_vars="Date", var_name="Ticker", value_name="AdjClose")
    tidy["Date"] = pd.to_datetime(tidy["Date"])
    tidy = tidy.dropna(subset=["AdjClose"])
    tidy["Volume"] = float("nan")
    return tidy[["Date", "Ticker", "AdjClose", "Volume"]]


def _read_existing_volumes() -> Optional[pd.DataFrame]:
    if not VOLUME_PATH.exists():
        return None
    df = pd.read_csv(VOLUME_PATH, parse_dates=["Date"])
    if {"Ticker", "Volume"}.issubset(df.columns):
        tidy = df[["Date", "Ticker", "Volume"]].copy()
        tidy["Date"] = pd.to_datetime(tidy["Date"])
        tidy["Volume"] = pd.to_numeric(tidy["Volume"], errors="coerce")
        return tidy
    if "Date" not in df.columns:
        LOGGER.warning("Existing volume file missing 'Date' column; ignoring cache at %s", VOLUME_PATH)
        return None
    value_cols = [col for col in df.columns if col != "Date"]
    if not value_cols:
        LOGGER.warning("Existing volume cache has no ticker columns; ignoring cache at %s", VOLUME_PATH)
        return None
    tidy = df.melt(id_vars="Date", var_name="Ticker", value_name="Volume")
    tidy["Date"] = pd.to_datetime(tidy["Date"])
    tidy["Volume"] = pd.to_numeric(tidy["Volume"], errors="coerce")
    return tidy


def _read_ticker_caches(
    tickers: Sequence[str], meta: Dict[str, Any]
) -> tuple[pd.DataFrame, Dict[str, str], bool]:
    frames: List[pd.DataFrame] = []
    hashes: Dict[str, str] = {}
    bucket = meta.get("tickers", {}) if isinstance(meta, dict) else {}
    changed = False
    for ticker in tickers:
        record = bucket.get(ticker) if isinstance(bucket, dict) else None
        cache_hash = record.get("data_hash") if isinstance(record, dict) else None
        cache_path = _find_existing_ticker_cache(ticker, cache_hash)
        if cache_path is None:
            continue
        try:
            frame = _read_ticker_cache(cache_path, ticker)
        except Exception as exc:  # pragma: no cover - best effort cache
            LOGGER.warning("Failed to read ticker cache for %s: %s", ticker, exc)
            continue
        if frame.empty:
            continue
        frames.append(frame)
        if cache_hash is None:
            cache_hash = hash_dataframe(frame[["Date", "AdjClose", "Volume"]])
            record = record or {}
            record["data_hash"] = cache_hash
            bucket[ticker] = record
            changed = True
        hashes[ticker] = cache_hash
    if frames:
        meta["tickers"] = bucket
        return _merge_frames(frames), hashes, changed
    return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"]), hashes, changed


@lru_cache(maxsize=1)
def _load_offline_table() -> pd.DataFrame:
    if os.getenv("OFFLINE_FALLBACK_PATH") is None:
        LOGGER.info("Using default offline fallback path at %s", OFFLINE_FALLBACK_PATH)
    if not OFFLINE_FALLBACK_PATH.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    try:
        table = pd.read_csv(OFFLINE_FALLBACK_PATH, parse_dates=["Date"])
    except Exception as exc:  # pragma: no cover - fallback best effort
        LOGGER.warning("Failed to load offline fallback at %s: %s", OFFLINE_FALLBACK_PATH, exc)
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    required = ["Date", "Ticker", "AdjClose"]
    if not set(required).issubset(table.columns):
        LOGGER.warning(
            "Offline fallback at %s missing required columns %s",
            OFFLINE_FALLBACK_PATH,
            set(required) - set(table.columns),
        )
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    has_volume = "Volume" in table.columns
    cols = list(required) + (["Volume"] if has_volume else [])
    table = table[cols].copy()
    if has_volume:
        table["Volume"] = pd.to_numeric(table["Volume"], errors="coerce")
    else:
        table["Volume"] = float("nan")
    table["Ticker"] = table["Ticker"].astype(str)
    table["AdjClose"] = pd.to_numeric(table["AdjClose"], errors="coerce")
    table = table.dropna(subset=["AdjClose"])
    return table


def _offline_prices_for(ticker: str, start: pd.Timestamp, end: dt.date) -> pd.DataFrame:
    table = _load_offline_table()
    if table.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    mask = (
        (table["Ticker"].str.upper() == ticker.upper())
        & (table["Date"] >= start)
        & (table["Date"] <= pd.Timestamp(end))
    )
    subset = table.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    subset["Ticker"] = ticker
    LOGGER.info(
        "Loaded %d offline rows for %s covering %s to %s",
        len(subset),
        ticker,
        subset["Date"].min().date(),
        subset["Date"].max().date(),
    )
    return subset


def _tidy_to_wide(df: pd.DataFrame, value_col: str = "AdjClose") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date"])
    if value_col not in df.columns:
        raise ValueError(f"Tidy frame missing expected value column '{value_col}'")
    pivot = (
        df.pivot_table(index="Date", columns="Ticker", values=value_col, aggfunc="last")
        .sort_index()
    )
    pivot = pivot.reset_index()
    pivot.columns = ["date", *[str(col) for col in pivot.columns[1:]]]
    return pivot


def _write_wide(df: pd.DataFrame, path: pathlib.Path, *, label: str) -> None:
    df = df.copy()
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.rename(columns={"date": "Date"})
    df.to_csv(path, index=False)
    LOGGER.info("%s cache updated at %s with %d rows", label, path, len(df))


def _ensure_meta_bucket(meta: Dict[str, Any]) -> Dict[str, Any]:
    bucket = meta.get("tickers")
    if not isinstance(bucket, dict):
        bucket = {}
        meta["tickers"] = bucket
    return bucket


def _read_cache_meta() -> Dict[str, Any]:
    if not CACHE_META_PATH.exists():
        return {"tickers": {}}
    try:
        with CACHE_META_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - best effort cache
        LOGGER.warning("Failed to read cache metadata at %s; recreating", CACHE_META_PATH)
        return {"tickers": {}}
    if not isinstance(data, dict):
        return {"tickers": {}}
    bucket = data.get("tickers")
    if not isinstance(bucket, dict):
        data["tickers"] = {}
    return data


def _write_cache_meta(meta: Dict[str, Any]) -> None:
    CACHE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def _update_meta_record(
    meta: Dict[str, Any],
    ticker: str,
    *,
    data_hash: Optional[str] = None,
    rows: Optional[int] = None,
    last_date: Optional[dt.date] = None,
    refreshed: bool,
    timestamp: Optional[dt.datetime] = None,
) -> None:
    bucket = _ensure_meta_bucket(meta)
    record = bucket.get(ticker, {})
    stamp = timestamp or dt.datetime.utcnow()
    record["last_attempt"] = stamp.isoformat()
    if refreshed:
        record["last_refresh"] = stamp.isoformat()
        record.pop("failures", None)
    else:
        record["failures"] = int(record.get("failures", 0)) + 1
    if data_hash is not None:
        record["data_hash"] = data_hash
    if rows is not None:
        record["rows"] = int(rows)
    if last_date is not None:
        record["last_date"] = last_date.isoformat()
    bucket[ticker] = record


def _should_skip_download(
    ticker: str,
    start: pd.Timestamp,
    today: dt.date,
    meta: Dict[str, Any],
    force_refresh: bool,
    cache_hash: Optional[str],
) -> bool:
    if force_refresh:
        return False
    if start >= pd.Timestamp(today):
        return True
    if not cache_hash:
        return False
    bucket = meta.get("tickers", {})
    record = bucket.get(ticker) if isinstance(bucket, dict) else None
    if not record:
        return False
    return record.get("data_hash") == cache_hash


def _resume_dates(
    tickers: Sequence[str],
    existing: Optional[pd.DataFrame],
    default_start: dt.date,
) -> Dict[str, pd.Timestamp]:
    resume: Dict[str, pd.Timestamp] = {}
    default_ts = pd.Timestamp(default_start)
    grouped = existing.groupby("Ticker") if existing is not None else None
    for ticker in tickers:
        if grouped is not None and ticker in grouped.groups:
            last_date = grouped.get_group(ticker)["Date"].max()
            resume[ticker] = pd.Timestamp(last_date) + pd.Timedelta(days=1)
        else:
            resume[ticker] = default_ts
    return resume


def _download_single_ticker(ticker: str, start: pd.Timestamp, end: dt.date) -> pd.DataFrame:
    if start >= pd.Timestamp(end):
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    proxy_url = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    if not _circuit_allows():
        LOGGER.warning("Circuit breaker open; using fallback provider for %s", ticker)
        return _download_fallback_provider(ticker, start, end)
    try:
        session = _get_requests_session()
        _rate_limit_pause()
        history = yf.download(
            ticker,
            start=start.to_pydatetime().date(),
            end=end + dt.timedelta(days=1),
            progress=False,
            auto_adjust=False,
            threads=False,
            session=session,
        )
    except Exception as exc:  # pragma: no cover - network/runtime errors
        LOGGER.warning("Failed to download %s: %s", ticker, exc)
        _circuit_record_failure()
        return _offline_prices_for(ticker, start, end)
    frame = _history_to_tidy_frame(history, ticker)
    if frame.empty:
        LOGGER.warning("No adjusted close data returned for %s", ticker)
        _circuit_record_failure()
        fallback = _offline_prices_for(ticker, start, end)
        if not fallback.empty:
            return fallback
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    _circuit_record_success()
    return frame


def _history_to_tidy_frame(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])

    price_series: Optional[pd.Series] = None
    for field in ("Adj Close", "Close"):
        if isinstance(history.columns, pd.MultiIndex):
            level0 = history.columns.get_level_values(0)
            if field not in level0:
                continue
            field_frame = history.xs(field, axis=1, level=0)
            if isinstance(field_frame, pd.Series):
                price_series = field_frame
            elif ticker in field_frame.columns:
                price_series = field_frame[ticker]
            elif len(field_frame.columns):
                price_series = field_frame.iloc[:, 0]
            else:
                price_series = None
            if price_series is not None:
                break
        else:
            if field in history.columns:
                series = history[field]
                if isinstance(series, pd.DataFrame):
                    price_series = series.iloc[:, 0]
                else:
                    price_series = series
                break

    if price_series is None:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])

    volume_series: Optional[pd.Series] = None
    if isinstance(history.columns, pd.MultiIndex):
        level0 = history.columns.get_level_values(0)
        if "Volume" in level0:
            volume_frame = history.xs("Volume", axis=1, level=0)
            if isinstance(volume_frame, pd.Series):
                volume_series = volume_frame
            elif ticker in volume_frame.columns:
                volume_series = volume_frame[ticker]
            elif len(volume_frame.columns):
                volume_series = volume_frame.iloc[:, 0]
    else:
        if "Volume" in history.columns:
            vol_col = history["Volume"]
            volume_series = vol_col.iloc[:, 0] if isinstance(vol_col, pd.DataFrame) else vol_col

    frame = price_series.to_frame(name="AdjClose").reset_index()
    if volume_series is not None:
        frame["Volume"] = volume_series.values
    else:
        frame["Volume"] = float("nan")
    date_col = frame.columns[0]
    frame = frame.rename(columns={date_col: "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["AdjClose"] = pd.to_numeric(frame["AdjClose"], errors="coerce")
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    frame["Ticker"] = ticker
    frame = frame.dropna(subset=["Date", "AdjClose"])
    return frame[["Date", "Ticker", "AdjClose", "Volume"]]


def _download_with_retry(
    ticker: str, start: pd.Timestamp, end: dt.date, *, max_attempts: int = MAX_RETRIES
) -> pd.DataFrame:
    delay = RETRY_BASE_DELAY
    for attempt in range(1, max_attempts + 1):
        frame = _download_single_ticker(ticker, start, end)
        if not frame.empty or start >= pd.Timestamp(end):
            return frame
        if attempt < max_attempts:
            time.sleep(delay)
            delay *= 2
    LOGGER.warning("No data returned for %s after %d attempts", ticker, max_attempts)
    return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])


def _merge_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose", "Volume"])
    merged = pd.concat(valid_frames, ignore_index=True)
    if "Volume" not in merged.columns:
        merged["Volume"] = float("nan")
    merged["AdjClose"] = pd.to_numeric(merged["AdjClose"], errors="coerce")
    merged["Volume"] = pd.to_numeric(merged["Volume"], errors="coerce")
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = (
        merged.dropna(subset=["Date", "AdjClose"])
        .drop_duplicates(subset=["Date", "Ticker"], keep="last")
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    return merged


def ensure_price_data(
    tickers: Optional[Sequence[str]] = None,
    force_refresh: bool = False,
    default_start: dt.date = DEFAULT_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure price+volume caches exist and return them in wide format.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(price_df, volume_df)`` with a ``date`` column followed by ticker
        columns in wide format.
    """

    _ensure_data_dir()
    start_time = time.perf_counter()
    ticker_list = sorted(set(tickers or ALL_TICKERS))
    if not ticker_list:
        raise ValueError("No tickers provided for download")

    download_mode = settings.vecm_price_download.strip().lower()
    if download_mode in _DOWNLOAD_FORCE:
        force_refresh = True
    offline_requested = download_mode in _DOWNLOAD_DISABLE and not force_refresh

    meta = _read_cache_meta() if not force_refresh else {"tickers": {}}
    today = dt.date.today()
    existing_hashes: Dict[str, str] = {}
    existing = None
    meta_changed = False
    if not force_refresh:
        ticker_cache, existing_hashes, cache_meta_changed = _read_ticker_caches(ticker_list, meta)
        if cache_meta_changed:
            meta_changed = True
        csv_cache = _read_existing_prices()
        volume_cache = _read_existing_volumes()
        if csv_cache is not None and volume_cache is not None:
            csv_cache = csv_cache.merge(volume_cache, on=["Date", "Ticker"], how="left")
            if "Volume_x" in csv_cache.columns:
                csv_cache["Volume"] = csv_cache["Volume_x"]
                csv_cache = csv_cache.drop(columns=["Volume_x"], errors="ignore")
            if "Volume_y" in csv_cache.columns:
                csv_cache["Volume"] = csv_cache["Volume_y"].combine_first(csv_cache.get("Volume"))
                csv_cache = csv_cache.drop(columns=["Volume_y"], errors="ignore")
        if not ticker_cache.empty and csv_cache is not None:
            existing = _merge_frames([csv_cache, ticker_cache])
        elif not ticker_cache.empty:
            existing = ticker_cache
        else:
            existing = csv_cache
    if existing is not None and not force_refresh:
        resume_dates = _resume_dates(ticker_list, existing, default_start)
        missing_tickers = sorted(set(ticker_list) - set(existing["Ticker"].unique()))
        wide_price = _tidy_to_wide(existing, value_col="AdjClose")
        current_hash = hash_dataframe(wide_price)
        cached_hash = meta.get("data_hash")
        if cached_hash == current_hash and not missing_tickers:
            LOGGER.info("Price cache unchanged (hash=%s); skipping download", current_hash[:12])
            wide_volume = _tidy_to_wide(existing, value_col="Volume")
            _write_wide(wide_price, DATA_PATH, label="Price")
            _write_wide(wide_volume, VOLUME_PATH, label="Volume")
            if meta_changed:
                _write_cache_meta(meta)
            return wide_price, wide_volume
    else:
        resume_dates = _resume_dates(ticker_list, existing, default_start)

    download_plan = []
    throttled: List[str] = []
    if offline_requested:
        LOGGER.info(
            "Price download disabled via %s; relying on cached data only",
            DOWNLOAD_CONTROL_ENV,
        )
        missing = set(ticker_list)
        if existing is not None:
            missing -= set(existing["Ticker"].unique())
        if missing:
            LOGGER.warning(
                "Offline mode active but cache lacks %d tickers: %s",
                len(missing),
                ", ".join(sorted(missing)[:10]),
            )
            if len(missing) > 10:
                LOGGER.warning("...and %d more", len(missing) - 10)
    else:
        for ticker in ticker_list:
            start = resume_dates[ticker]
            cache_hash = existing_hashes.get(ticker)
            if _should_skip_download(ticker, start, today, meta, force_refresh, cache_hash):
                throttled.append(ticker)
                continue
            download_plan.append((ticker, start))

    if throttled:
        LOGGER.info("Skipping %d tickers with unchanged cache hashes", len(throttled))

    new_frames: List[pd.DataFrame] = []
    refreshed_tickers: List[str] = []
    failed_tickers: List[str] = []
    if download_plan:
        max_workers = max(1, min(MAX_WORKERS, len(download_plan)))
        LOGGER.info(
            "Downloading %d tickers in %d workers (force_refresh=%s)",
            len(download_plan),
            max_workers,
            force_refresh,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_download_with_retry, ticker, start, today): ticker
                for ticker, start in download_plan
            }
            for future in concurrent.futures.as_completed(future_map):
                ticker = future_map[future]
                try:
                    frame = future.result()
                except Exception as exc:  # pragma: no cover - network/runtime errors
                    LOGGER.warning("Download task failed for %s: %s", ticker, exc)
                    _update_meta_record(meta, ticker, refreshed=False)
                    failed_tickers.append(ticker)
                    meta_changed = True
                    continue
                if not frame.empty:
                    new_frames.append(frame)
                    refreshed_tickers.append(ticker)
                    _update_meta_record(meta, ticker, refreshed=True)
                else:
                    _update_meta_record(meta, ticker, refreshed=False)
                    failed_tickers.append(ticker)
                meta_changed = True
    else:
        LOGGER.info("No tickers required download; cache is up to date")

    combined = _merge_frames([existing, _merge_frames(new_frames)])
    if combined.empty:
        raise FileNotFoundError(
            "Streaming download did not yield any adjusted close data. Check ticker list or network connectivity."
        )

    meta_bucket = _ensure_meta_bucket(meta)
    for ticker in ticker_list:
        ticker_frame = combined.loc[combined["Ticker"] == ticker, ["Date", "AdjClose", "Volume"]].copy()
        if ticker_frame.empty:
            continue
        ticker_frame = ticker_frame.sort_values("Date")
        data_hash = hash_dataframe(ticker_frame[["Date", "AdjClose", "Volume"]])
        record = meta_bucket.get(ticker, {}) if isinstance(meta_bucket, dict) else {}
        cached_hash = record.get("data_hash")
        cache_path = _find_existing_ticker_cache(ticker, cached_hash)
        if cached_hash == data_hash and cache_path is not None and cache_path.exists():
            _update_meta_record(
                meta,
                ticker,
                data_hash=data_hash,
                rows=len(ticker_frame),
                last_date=ticker_frame["Date"].max().date(),
                refreshed=True,
            )
            continue
        data_hash = _write_ticker_cache(ticker, ticker_frame.assign(Ticker=ticker))
        _update_meta_record(
            meta,
            ticker,
            data_hash=data_hash,
            rows=len(ticker_frame),
            last_date=ticker_frame["Date"].max().date(),
            refreshed=True,
        )
        meta_changed = True

    wide_price = _tidy_to_wide(combined, value_col="AdjClose")
    wide_volume = _tidy_to_wide(combined, value_col="Volume")
    new_hash = hash_dataframe(wide_price)
    if meta.get("data_hash") != new_hash:
        meta_changed = True
    meta["data_hash"] = new_hash
    _write_wide(wide_price, DATA_PATH, label="Price")
    _write_wide(wide_volume, VOLUME_PATH, label="Volume")
    if refreshed_tickers:
        LOGGER.info("Refreshed %d tickers: %s", len(refreshed_tickers), ", ".join(sorted(refreshed_tickers)[:10]))
        if len(refreshed_tickers) > 10:
            LOGGER.info("...and %d more", len(refreshed_tickers) - 10)
    if failed_tickers:
        LOGGER.warning(
            "Failed to refresh %d tickers: %s",
            len(failed_tickers),
            ", ".join(sorted(failed_tickers)[:10]),
        )
        if len(failed_tickers) > 10:
            LOGGER.warning("...and %d more", len(failed_tickers) - 10)
    duration_s = time.perf_counter() - start_time
    _write_download_metrics(
        refreshed=refreshed_tickers,
        failed=failed_tickers,
        throttled=throttled,
        duration_s=duration_s,
    )
    if meta_changed:
        _write_cache_meta(meta)
    return wide_price, wide_volume


def load_cached_prices(path: Optional[pathlib.Path | str] = None) -> pd.DataFrame:
    """Load the cached adjusted close prices in wide format without downloading."""

    cache_path = pathlib.Path(path or DATA_PATH)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Price cache not found at {cache_path}. Run ensure_price_data() to populate it first."
        )
    df = pd.read_csv(cache_path, parse_dates=["Date"])
    if {"Ticker", "AdjClose"}.issubset(df.columns):
        df = _tidy_to_wide(df[["Date", "Ticker", "AdjClose"]])
    elif "Date" not in df.columns:
        raise ValueError(f"Price cache at {cache_path} is missing a 'Date' column")
    else:
        df = df.sort_values("Date")
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns:
        raise ValueError(f"Price cache at {cache_path} does not expose a date column")
    return df


def load_cached_volumes(path: Optional[pathlib.Path | str] = None) -> pd.DataFrame:
    """Load cached volume data in wide format without downloading."""

    cache_path = pathlib.Path(path or VOLUME_PATH)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Volume cache not found at {cache_path}. Run ensure_price_data() to populate it first."
        )
    df = pd.read_csv(cache_path, parse_dates=["Date"])
    if {"Ticker", "Volume"}.issubset(df.columns):
        df = _tidy_to_wide(df[["Date", "Ticker", "Volume"]], value_col="Volume")
    elif "Date" not in df.columns:
        raise ValueError(f"Volume cache at {cache_path} is missing a 'Date' column")
    else:
        df = df.sort_values("Date")
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns:
        raise ValueError(f"Volume cache at {cache_path} does not expose a date column")
    return df


if __name__ == "__main__":  # pragma: no cover - smoke test helper
    try:
        recent_start = dt.date.today() - dt.timedelta(days=30)
        price_df, volume_df = ensure_price_data(tickers=["^JKSE", "USDIDR=X"], default_start=recent_start)
        LOGGER.info(
            "Downloaded %d price rows and %d volume rows across %d tickers",
            len(price_df),
            len(volume_df),
            max(0, len(price_df.columns) - 1),
        )
    except Exception as exc:
        LOGGER.error("Data streaming smoke test failed: %s", exc)
