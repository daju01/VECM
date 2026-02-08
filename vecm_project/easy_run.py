from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vecm_project.scripts import playbook_vecm

BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "config" / "strategy_profiles.json"


def _load_profiles(path: Path = PROFILE_PATH) -> Dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("strategy_profiles.json must be a JSON object")
    return payload


def _build_params(profile: Mapping[str, Any], pair: str, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    params = {}
    params.update(profile.get("params", {}))
    params["subset"] = pair
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VECM playbook using strategy profiles.")
    parser.add_argument("--profile", default="beginner", help="Profile name (default: beginner)")
    parser.add_argument("--pair", required=True, help='Pair ticker, e.g. "BBCA.JK,BBRI.JK"')
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print merged config without running playbook")
    parser.add_argument("--horizon", default=None, help="Override horizon name (optional)")
    parser.add_argument("--method", default=None, help="Override method (optional)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profiles = _load_profiles()
    if args.list_profiles:
        for name, data in profiles.items():
            label = data.get("label", name)
            desc = data.get("description", "")
            print(f"{name}: {label} - {desc}")
        return
    if args.profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown profile '{args.profile}'. Available: {available}")

    overrides: Dict[str, Optional[str]] = {"method": args.method, "horizon": args.horizon}
    config = _build_params(profiles[args.profile], args.pair, overrides)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    playbook_vecm.run_playbook(config, persist=True)


if __name__ == "__main__":
    main()
