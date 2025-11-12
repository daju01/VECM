"""Streaming adjusted close data loader for Indonesian equities universe."""
from __future__ import annotations

import datetime as dt
import pathlib
import time
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import yfinance as yf

from . import storage

LOGGER = storage.configure_logging("data_streaming")
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "adj_close_data.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_adj_close.csv"
DEFAULT_START_DATE = dt.date(2013, 1, 1)


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


def _tidy_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date"])
    pivot = (
        df.pivot_table(index="Date", columns="Ticker", values="AdjClose", aggfunc="last")
        .sort_index()
    )
    pivot = pivot.reset_index()
    pivot.columns = ["Date", *[str(col) for col in pivot.columns[1:]]]
    return pivot


def _write_wide(df: pd.DataFrame) -> None:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.to_csv(DATA_PATH, index=False)
    LOGGER.info("Price cache updated at %s with %d rows", DATA_PATH, len(df))


def _load_sample_prices() -> pd.DataFrame:
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(
            "No bundled sample price data is available; unable to recover from download failure.",
        )
    sample = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"])
    if "Date" not in sample.columns or len(sample.columns) < 3:
        raise ValueError(
            f"Bundled sample price data at {SAMPLE_DATA_PATH} is malformed; expected Date plus >=2 tickers."
        )
    sample = sample.sort_values("Date")
    LOGGER.warning(
        "Falling back to bundled sample prices at %s with %d rows and %d tickers",
        SAMPLE_DATA_PATH,
        len(sample),
        len(sample.columns) - 1,
    )
    return sample


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
    try:
        history = yf.download(
            ticker,
            start=start.to_pydatetime().date(),
            end=end + dt.timedelta(days=1),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network/runtime errors
        LOGGER.warning("Failed to download %s: %s", ticker, exc)
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    if history.empty or "Adj Close" not in history.columns:
        LOGGER.warning("No adjusted close data returned for %s", ticker)
        return pd.DataFrame(columns=["Date", "Ticker", "AdjClose"])
    frame = history.reset_index()[["Date", "Adj Close"]].rename(columns={"Adj Close": "AdjClose"})
    frame["Ticker"] = ticker
    return frame


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

    existing = None if force_refresh else _read_existing_prices()
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

    new_frames: List[pd.DataFrame] = []
    for idx, ticker in enumerate(ticker_list, start=1):
        start = resume_dates[ticker]
        frame = _download_single_ticker(ticker, start, today)
        if not frame.empty:
            new_frames.append(frame)
        if idx % 10 == 0:
            time.sleep(0.25)

    combined = _merge_frames([existing, _merge_frames(new_frames)])
    if combined.empty:
        sample = _load_sample_prices()
        _write_wide(sample)
        return sample

    wide = _tidy_to_wide(combined)
    _write_wide(wide)
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
    df = df.rename(columns={"Date": "date"})
    return df


if __name__ == "__main__":  # pragma: no cover - smoke test helper
    try:
        recent_start = dt.date.today() - dt.timedelta(days=30)
        df = ensure_price_data(tickers=["^JKSE", "USDIDR=X"], default_start=recent_start)
        LOGGER.info("Downloaded %d rows across %d tickers", len(df), df["Ticker"].nunique())
    except Exception as exc:
        LOGGER.error("Data streaming smoke test failed: %s", exc)
