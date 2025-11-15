"""Streaming adjusted close data loader for Indonesian equities universe."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import time
from functools import lru_cache
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

from . import storage

LOGGER = storage.configure_logging("data_streaming")
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "adj_close_data.csv"
CACHE_META_PATH = DATA_DIR / "adj_close_data.meta.json"
DEFAULT_START_DATE = dt.date(2013, 1, 1)
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.75
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


# ---------------------------------------------------------------------------
# Ticker universe replicated from the R reference script
# ---------------------------------------------------------------------------
MACRO_TICKERS = ["^JKSE", "USDIDR=X"]
COMMODITY_PRICE = ["GC=F"]

BANKING_GROUP = [
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
]

TELCO_GROUP = ["TLKM.JK", "ISAT.JK", "EXCL.JK", "FREN.JK", "MTEL.JK", "TOWR.JK", "TBIG.JK"]

DATACENTER_GROUP = ["DCII.JK", "EDGE.JK"]

DIGITAL_PLATFORM_GROUP = [
    "BUKA.JK",
    "GOTO.JK",
    "EMTK.JK",
    "MCAS.JK",
    "SCMA.JK",
    "TECH.JK",
    "BELI.JK",
]

COAL_GROUP = [
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
]

MINERAL_GROUP = ["TINS.JK", "BRMS.JK", "NICL.JK"]

STRATEGIC_RESOURCE_GROUP = [
    "MDKA.JK",
    "AMMN.JK",
    "ANTM.JK",
    "INCO.JK",
    "MBMA.JK",
    "NCKL.JK",
]

ENERGY_CHEMICAL_GROUP = ["ESSA.JK", "RAJA.JK", "BRPT.JK", "BREN.JK", "DSSA.JK", "PGAS.JK", "INDY.JK"]

FOOD_BEVERAGE_GROUP = ["ICBP.JK", "INDF.JK", "MYOR.JK", "ULTJ.JK", "CMRY.JK", "CLEO.JK", "ROTI.JK", "JPFA.JK"]

NON_FOOD_CONSUMER_GROUP = ["UNVR.JK", "HMSP.JK", "GGRM.JK", "KLBF.JK", "SIDO.JK", "WIIM.JK"]

INDUSTRY_INFRA_GROUP = [
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
]

INVESTMENT_GROUP = ["SRTG.JK", "MIDI.JK", "SCMA.JK", "FILM.JK", "MNCN.JK", "LPPF.JK"]

PETROCHEMICAL_GROUP = ["TPIA.JK", "BRPT.JK"]

PALMOIL_GROUP = [
    "AALI.JK",
    "LSIP.JK",
    "SIMP.JK",
    "DSNG.JK",
    "SMAR.JK",
    "TBLA.JK",
    "BWPT.JK",
    "TAPG.JK",
    "PALM.JK",
]

MULTIFINANCE_GROUP = ["ADMF.JK", "BFIN.JK", "MFIN.JK", "IMJS.JK", "TIFA.JK", "CFIN.JK", "TFCM.JK", "BCAP.JK"]

MEDIA_RETAIL_GROUP = ["SCMA.JK", "MNCN.JK", "FILM.JK", "LPPF.JK"]

CONSUMER_RETAIL_GROUP = ["AMRT.JK", "MDIY.JK", "DNET.JK"]

PROPERTY_GROUP = [
    "PANI.JK",
    "BSDE.JK",
    "CTRA.JK",
    "SMRA.JK",
    "ASRI.JK",
    "BKSL.JK",
    "DMAS.JK",
    "LPKR.JK",
    "LCGP.JK",
]

HEALTHCARE_PROVIDERS = ["MIKA.JK", "PRDA.JK", "SAME.JK", "SILO.JK", "HEAL.JK"]
LOGISTICS_TRANSPORT = ["ASSA.JK", "BIRD.JK"]
SHIPPING = ["SMDR.JK"]
AVIATION = ["GIAA.JK"]
CEMENT_MATERIALS = ["INTP.JK"]
SPECIALTY_RETAIL = ["RALS.JK", "ERAA.JK", "MAPI.JK", "ACES.JK"]
POULTRY_FEED = ["MAIN.JK", "CPIN.JK"]
BANKING_ADDITIONAL = ["BTPS.JK", "BTPN.JK"]
INDUSTRIAL_PROPERTY = ["SSIA.JK", "BEST.JK"]
PULP_PAPER = ["INKP.JK", "TKIM.JK"]
ENERGY_RESOURCES = ["MEDC.JK"]
RENEWABLES_UTILITIES = ["KEEN.JK", "ARKO.JK", "PGEO.JK"]

ALL_TICKERS = sorted(
    set(
        MACRO_TICKERS
        + COMMODITY_PRICE
        + BANKING_GROUP
        + TELCO_GROUP
        + DATACENTER_GROUP
        + DIGITAL_PLATFORM_GROUP
        + COAL_GROUP
        + MINERAL_GROUP
        + STRATEGIC_RESOURCE_GROUP
        + ENERGY_CHEMICAL_GROUP
        + FOOD_BEVERAGE_GROUP
        + NON_FOOD_CONSUMER_GROUP
        + INDUSTRY_INFRA_GROUP
        + INVESTMENT_GROUP
        + PETROCHEMICAL_GROUP
        + PALMOIL_GROUP
        + MULTIFINANCE_GROUP
        + MEDIA_RETAIL_GROUP
        + CONSUMER_RETAIL_GROUP
        + PROPERTY_GROUP
        + HEALTHCARE_PROVIDERS
        + LOGISTICS_TRANSPORT
        + SHIPPING
        + AVIATION
        + CEMENT_MATERIALS
        + SPECIALTY_RETAIL
        + POULTRY_FEED
        + BANKING_ADDITIONAL
        + INDUSTRIAL_PROPERTY
        + PULP_PAPER
        + ENERGY_RESOURCES
        + RENEWABLES_UTILITIES
    )
)


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_requests_session() -> Any:
    global _REQUESTS_SESSION
    if _REQUESTS_SESSION is not None:
        return _REQUESTS_SESSION

    user_agent = os.getenv(YF_USER_AGENT_ENV, "").strip() or YF_DEFAULT_USER_AGENT
    impersonate = os.getenv(YF_IMPERSONATE_ENV, "chrome124").strip() or "chrome124"
    verify_path = os.getenv(YF_VERIFY_ENV, "").strip() or os.getenv("CODEX_PROXY_CERT", "").strip()
    proxy_auth = os.getenv(YF_PROXY_AUTH_ENV, "").strip()
    proxies = get_environ_proxies("https://query1.finance.yahoo.com")

    if curl_requests is not None:
        session = curl_requests.Session(impersonate=impersonate, trust_env=False)
        session.headers.setdefault("User-Agent", user_agent)
        session.headers.setdefault(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        session.headers.setdefault("Accept-Language", "en-US,en;q=0.9")
        session.headers.setdefault("Host", "query1.finance.yahoo.com")
        session.headers.setdefault("Connection", "keep-alive")
        session.headers.setdefault("Proxy-Connection", "keep-alive")
        session.headers.setdefault("Accept-Encoding", "gzip, deflate, br")
        session.timeout = 30
    else:
        session = requests.Session()
        session.headers.setdefault("User-Agent", user_agent)
        session.headers.setdefault(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        session.headers.setdefault("Accept-Language", "en-US,en;q=0.9")
        session.headers.setdefault("Host", "query1.finance.yahoo.com")
        session.headers.setdefault("Connection", "keep-alive")
        session.headers.setdefault("Proxy-Connection", "keep-alive")
        session.headers.setdefault("Accept-Encoding", "gzip, deflate, br")
        session.trust_env = False
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

    session.trust_env = False

    if proxies:
        session.proxies = proxies
        if proxy_auth:
            session.headers.setdefault("Proxy-Authorization", proxy_auth)
    else:
        session.proxies = {}

    if verify_path:
        session.verify = verify_path

    _REQUESTS_SESSION = session
    return session


def _read_existing_prices() -> Optional[pd.DataFrame]:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    if {"Ticker", "AdjClose"}.issubset(df.columns):
        tidy = df[["Date", "Ticker", "AdjClose"]].copy()
        tidy["Date"] = pd.to_datetime(tidy["Date"])
        return tidy
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
    return tidy[["Date", "Ticker", "AdjClose"]]


@lru_cache(maxsize=1)
def _load_offline_table() -> pd.DataFrame:
    if not OFFLINE_FALLBACK_PATH.exists():
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    try:
        table = pd.read_csv(OFFLINE_FALLBACK_PATH, parse_dates=["Date"])
    except Exception as exc:  # pragma: no cover - fallback best effort
        LOGGER.warning("Failed to load offline fallback at %s: %s", OFFLINE_FALLBACK_PATH, exc)
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    required = {"Date", "Ticker", "AdjClose"}
    if not required.issubset(table.columns):
        LOGGER.warning(
            "Offline fallback at %s missing required columns %s", OFFLINE_FALLBACK_PATH, required - set(table.columns)
        )
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    table = table[list(required)].copy()
    table["Ticker"] = table["Ticker"].astype(str)
    table["AdjClose"] = pd.to_numeric(table["AdjClose"], errors="coerce")
    table = table.dropna(subset=["AdjClose"])
    return table


def _offline_prices_for(ticker: str, start: pd.Timestamp, end: dt.date) -> pd.DataFrame:
    table = _load_offline_table()
    if table.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    mask = (
        (table["Ticker"].str.upper() == ticker.upper())
        & (table["Date"] >= start)
        & (table["Date"] <= pd.Timestamp(end))
    )
    subset = table.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    subset["Ticker"] = ticker
    LOGGER.info(
        "Loaded %d offline rows for %s covering %s to %s",
        len(subset),
        ticker,
        subset["Date"].min().date(),
        subset["Date"].max().date(),
    )
    return subset


def _tidy_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date"])
    pivot = (
        df.pivot_table(index="Date", columns="Ticker", values="AdjClose", aggfunc="last")
        .sort_index()
    )
    pivot = pivot.reset_index()
    pivot.columns = ["date", *[str(col) for col in pivot.columns[1:]]]
    return pivot


def _write_wide(df: pd.DataFrame) -> None:
    df = df.copy()
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.rename(columns={"date": "Date"})
    df.to_csv(DATA_PATH, index=False)
    LOGGER.info("Price cache updated at %s with %d rows", DATA_PATH, len(df))


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
    meta: Dict[str, Any], ticker: str, *, refreshed: bool, timestamp: Optional[dt.datetime] = None
) -> None:
    bucket = _ensure_meta_bucket(meta)
    record = bucket.get(ticker, {})
    stamp = timestamp or dt.datetime.utcnow()
    record["last_attempt"] = stamp.isoformat()
    if refreshed:
        record["last_refresh"] = stamp.date().isoformat()
        record.pop("failures", None)
    else:
        record["failures"] = int(record.get("failures", 0)) + 1
    bucket[ticker] = record


def _should_skip_download(
    ticker: str,
    start: pd.Timestamp,
    today: dt.date,
    meta: Dict[str, Any],
    force_refresh: bool,
) -> bool:
    if force_refresh:
        return False
    if start >= pd.Timestamp(today):
        return True
    bucket = meta.get("tickers", {})
    record = bucket.get(ticker) if isinstance(bucket, dict) else None
    if not record:
        return False
    last_refresh = record.get("last_refresh")
    if not last_refresh:
        return False
    try:
        last_refresh_date = dt.date.fromisoformat(last_refresh)
    except ValueError:
        return False
    return last_refresh_date >= today


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
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    proxy_url = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    try:
        session = _get_requests_session()
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
        return _offline_prices_for(ticker, start, end)
    if history.empty or "Adj Close" not in history.columns:
        LOGGER.warning("No adjusted close data returned for %s", ticker)
        fallback = _offline_prices_for(ticker, start, end)
        if not fallback.empty:
            return fallback
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    frame = history.reset_index()[["Date", "Adj Close"]].rename(columns={"Adj Close": "AdjClose"})
    frame["Ticker"] = ticker
    return frame


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
    return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])


def _merge_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    merged = pd.concat(valid_frames, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = (
        merged.drop_duplicates(subset=["Date", "Ticker"], keep="last")
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    return merged


def ensure_price_data(
    tickers: Optional[Sequence[str]] = None,
    force_refresh: bool = False,
    default_start: dt.date = DEFAULT_START_DATE,
) -> pd.DataFrame:
    """Ensure the adjusted close cache exists and return it in wide format."""

    _ensure_data_dir()
    ticker_list = sorted(set(tickers or ALL_TICKERS))
    if not ticker_list:
        raise ValueError("No tickers provided for download")

    download_mode = os.getenv(DOWNLOAD_CONTROL_ENV, "auto").strip().lower()
    if download_mode in _DOWNLOAD_FORCE:
        force_refresh = True
    offline_requested = download_mode in _DOWNLOAD_DISABLE and not force_refresh

    existing = None if force_refresh else _read_existing_prices()
    meta = _read_cache_meta() if not force_refresh else {"tickers": {}}
    today = dt.date.today()
    if existing is not None and not force_refresh:
        resume_dates = _resume_dates(ticker_list, existing, default_start)
        missing_tickers = sorted(set(ticker_list) - set(existing["Ticker"].unique()))
        last_available = existing["Date"].max().date() if not existing.empty else DEFAULT_START_DATE
        within_one_day = last_available >= today - dt.timedelta(days=1)
        needs_refresh = any(start.date() <= today for start in resume_dates.values())
        updated_today = False
        try:
            updated_today = dt.date.fromtimestamp(DATA_PATH.stat().st_mtime) == today
        except OSError:
            updated_today = False
        if (updated_today or (within_one_day and not needs_refresh)) and not missing_tickers:
            LOGGER.info("Price cache already up to date at %s; skipping download", DATA_PATH)
            wide = _tidy_to_wide(existing)
            _write_wide(wide)
            return wide
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
            if _should_skip_download(ticker, start, today, meta, force_refresh):
                throttled.append(ticker)
                continue
            download_plan.append((ticker, start))

    if throttled:
        LOGGER.info("Skipping %d tickers already refreshed today", len(throttled))

    new_frames: List[pd.DataFrame] = []
    refreshed_tickers: List[str] = []
    meta_changed = False
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
                    meta_changed = True
                    continue
                if not frame.empty:
                    new_frames.append(frame)
                    refreshed_tickers.append(ticker)
                    _update_meta_record(meta, ticker, refreshed=True)
                else:
                    _update_meta_record(meta, ticker, refreshed=False)
                meta_changed = True
    else:
        LOGGER.info("No tickers required download; cache is up to date")

    if meta_changed:
        _write_cache_meta(meta)

    combined = _merge_frames([existing, _merge_frames(new_frames)])
    if combined.empty:
        raise FileNotFoundError(
            "Streaming download did not yield any adjusted close data. Check ticker list or network connectivity."
        )

    wide = _tidy_to_wide(combined)
    _write_wide(wide)
    if refreshed_tickers:
        LOGGER.info("Refreshed %d tickers: %s", len(refreshed_tickers), ", ".join(sorted(refreshed_tickers)[:10]))
        if len(refreshed_tickers) > 10:
            LOGGER.info("...and %d more", len(refreshed_tickers) - 10)
    return wide


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


if __name__ == "__main__":  # pragma: no cover - smoke test helper
    try:
        recent_start = dt.date.today() - dt.timedelta(days=30)
        df = ensure_price_data(tickers=["^JKSE", "USDIDR=X"], default_start=recent_start)
        LOGGER.info("Downloaded %d rows across %d tickers", len(df), df["Ticker"].nunique())
    except Exception as exc:
        LOGGER.error("Data streaming smoke test failed: %s", exc)
