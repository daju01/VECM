from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import smtplib
import ssl
import urllib.parse
import urllib.request
from email.message import EmailMessage
from typing import Any, Dict, Iterable, List, Mapping, Optional

LOGGER = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "daily"
DEFAULT_STATE_PATH = DEFAULT_OUTPUT_DIR / "notify_state.json"


def _read_json(path: pathlib.Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    raise ValueError(f"Unsupported JSON payload in {path}")


def _read_csv(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_signal(item: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
    return {
        "pair": item.get("pair") or item.get("subset"),
        "direction": str(item.get("direction") or "").upper() or "FLAT",
        "confidence": _coerce_float(item.get("confidence")),
        "expected_holding_period": _coerce_float(item.get("expected_holding_period")),
        "z_score": _coerce_float(item.get("z_score") or metrics.get("z_score")),
        "regime": _coerce_float(item.get("regime") or metrics.get("regime")),
        "timestamp": item.get("timestamp"),
    }


def _format_value(value: Optional[float], precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _format_message(signals: Iterable[Mapping[str, Any]]) -> str:
    lines = ["VECM Daily Signal Update"]
    for signal in signals:
        pair = signal.get("pair") or "(unknown)"
        direction = signal.get("direction") or "FLAT"
        confidence = _format_value(signal.get("confidence"))
        hold = _format_value(signal.get("expected_holding_period"))
        z_score = _format_value(signal.get("z_score"))
        regime = _format_value(signal.get("regime"))
        lines.append(
            f"- {pair}: {direction} | confidence {confidence} | hold {hold} days | "
            f"z-score {z_score}, regime {regime}"
        )
    return "\n".join(lines)


def _signal_signature(signal: Mapping[str, Any]) -> Dict[str, Any]:
    def norm_float(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(float(value), 4)

    return {
        "direction": signal.get("direction"),
        "confidence": norm_float(signal.get("confidence")),
        "z_score": norm_float(signal.get("z_score")),
        "regime": norm_float(signal.get("regime")),
    }


def _load_state(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: pathlib.Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(url, data=payload)
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status >= 400:
            raise RuntimeError(f"Telegram API error {response.status}")


def _send_email(message: str, subject: str) -> None:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    to_raw = os.getenv("SMTP_TO")
    if not host or not user or not password or not to_raw:
        raise RuntimeError("Missing SMTP_HOST/SMTP_USER/SMTP_PASS/SMTP_TO")
    port = int(os.getenv("SMTP_PORT", "587"))
    use_tls = os.getenv("SMTP_STARTTLS", "true").lower() not in {"0", "false", "no"}
    sender = os.getenv("SMTP_FROM", user)
    recipients = [addr.strip() for addr in to_raw.split(",") if addr.strip()]
    if not recipients:
        raise RuntimeError("SMTP_TO must contain at least one recipient")

    email = EmailMessage()
    email["Subject"] = subject
    email["From"] = sender
    email["To"] = ", ".join(recipients)
    email.set_content(message)

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as smtp:
        if use_tls:
            smtp.starttls(context=context)
        smtp.login(user, password)
        smtp.send_message(email)


def _resolve_input(path: Optional[str]) -> pathlib.Path:
    if path:
        return pathlib.Path(path).expanduser()
    candidates = sorted(DEFAULT_OUTPUT_DIR.glob("daily_signal_*.json"))
    if candidates:
        return candidates[-1]
    candidates = sorted(DEFAULT_OUTPUT_DIR.glob("daily_signal_*.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No daily_signal output found; run daily_signal.py first")


def _load_signals(path: pathlib.Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        raw = _read_json(path)
    elif path.suffix.lower() == ".csv":
        raw = _read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    return [_normalize_signal(item) for item in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send daily signal notifications via Telegram/email.")
    parser.add_argument("--input", help="Path to daily_signal JSON/CSV output")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH), help="Path to state file")
    parser.add_argument("--only-changed", action="store_true", default=True)
    parser.add_argument("--no-only-changed", dest="only_changed", action="store_false")
    parser.add_argument(
        "--channels",
        default="auto",
        choices=["auto", "telegram", "email", "both"],
        help="Notification channels (default: auto based on env vars)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    input_path = _resolve_input(args.input)
    signals = _load_signals(input_path)

    state_path = pathlib.Path(args.state_path).expanduser()
    state = _load_state(state_path) if args.only_changed else {}
    changed: List[Dict[str, Any]] = []
    next_state: Dict[str, Any] = {}

    for signal in signals:
        pair = signal.get("pair")
        if not pair:
            continue
        signature = _signal_signature(signal)
        previous = state.get(pair)
        if not args.only_changed or previous != signature:
            changed.append(signal)
        next_state[pair] = signature

    if args.only_changed:
        _save_state(state_path, next_state)

    if not changed:
        LOGGER.info("No signal changes detected; skipping notifications.")
        return

    message = _format_message(changed)
    subject = "VECM Daily Signal Update"

    telegram_ready = os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")
    email_ready = (
        os.getenv("SMTP_HOST")
        and os.getenv("SMTP_USER")
        and os.getenv("SMTP_PASS")
        and os.getenv("SMTP_TO")
    )

    channels = args.channels
    if channels == "auto":
        if telegram_ready and email_ready:
            channels = "both"
        elif telegram_ready:
            channels = "telegram"
        elif email_ready:
            channels = "email"
        else:
            raise RuntimeError("No notification channels configured; set Telegram or SMTP env vars.")

    if channels in {"telegram", "both"}:
        _send_telegram(message)
        LOGGER.info("Telegram notification sent.")
    if channels in {"email", "both"}:
        _send_email(message, subject)
        LOGGER.info("Email notification sent.")


if __name__ == "__main__":
    main()
