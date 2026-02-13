"""Scan parameter space for extreme full-sample Sharpe on a fixed pair."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vecm_project.scripts import playbook_vecm
from vecm_project.scripts.playbook_types import DecisionParams, FeatureBuildResult, PlaybookConfig

OUT_DIR = Path("vecm_project/out/bo")
MODE_FULL_SAMPLE = "full_sample"


@dataclass(frozen=True)
class Candidate:
    focus: str
    long_only: bool
    short_filter: bool
    gate_enforce: bool
    z_entry: float
    z_exit: float
    max_hold: int
    cooldown: int
    p_th: float


@dataclass(frozen=True)
class ScanArtifacts:
    candidates_path: Path
    summary_path: Path
    total_runs: int
    valid_runs: int


def compute_sharpe_full(ret_full: pd.Series, ann_days: int) -> float:
    """Compute full-sample Sharpe. Returns 0.0 for degenerate inputs."""
    if ret_full is None:
        return 0.0
    series = ret_full.dropna()
    if series.empty:
        return 0.0
    mu = float(series.mean())
    sd = float(series.std())
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        return 0.0
    ann = float(ann_days) if ann_days and ann_days > 0 else 252.0
    return float((mu / sd) * math.sqrt(ann))


def filter_valid_candidates(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Keep candidates eligible for extreme ranking."""
    if df.empty:
        return df.copy()
    mask = (
        (df["n_trades"].fillna(0).astype(int) >= int(min_trades))
        & np.isfinite(df["maxdd"].to_numpy(dtype=float))
        & np.isfinite(df["sharpe_full"].to_numpy(dtype=float))
    )
    return df.loc[mask].copy()


def rank_extremes(valid_df: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top positive and top negative full-sample Sharpe candidates."""
    if valid_df.empty:
        empty = valid_df.copy()
        return empty, empty
    k = max(int(top_k), 1)
    top_pos = valid_df.sort_values("sharpe_full", ascending=False).head(k).copy()
    top_neg = valid_df.sort_values("sharpe_full", ascending=True).head(k).copy()
    return top_pos, top_neg


def _normalize_pair(value: str) -> str:
    compact = str(value).strip().replace(" ", "")
    if not compact:
        raise ValueError("pair must not be empty")
    if "~" in compact:
        tokens = [token for token in compact.split("~") if token]
    else:
        tokens = [token for token in compact.split(",") if token]
    if len(tokens) != 2:
        raise ValueError("pair must be in format AAA,BBB or AAA~BBB")
    return f"{tokens[0]},{tokens[1]}"


def _sample_bool(rng: np.random.Generator, p_true: float) -> bool:
    return bool(rng.random() < p_true)


def _sample_z_pair(
    rng: np.random.Generator,
    *,
    z_entry_min: float,
    z_entry_max: float,
    z_exit_min: float,
    z_exit_max: float,
    min_gap: float,
) -> Tuple[float, float]:
    z_exit = float(rng.uniform(z_exit_min, z_exit_max))
    entry_low = max(z_entry_min, z_exit + min_gap)
    if entry_low > z_entry_max:
        z_exit = max(z_exit_min, z_entry_max - min_gap)
        entry_low = max(z_entry_min, z_exit + min_gap)
    z_entry = float(rng.uniform(entry_low, z_entry_max))
    return z_entry, z_exit


def _sample_candidate(rng: np.random.Generator, focus: str) -> Candidate:
    min_gap = 0.05
    if focus == "uniform":
        z_entry, z_exit = _sample_z_pair(
            rng,
            z_entry_min=0.50,
            z_entry_max=2.50,
            z_exit_min=0.20,
            z_exit_max=1.50,
            min_gap=min_gap,
        )
        long_only = _sample_bool(rng, 0.5)
        short_filter = _sample_bool(rng, 0.5)
        gate_enforce = _sample_bool(rng, 0.5)
        max_hold = int(rng.integers(1, 61))
        cooldown = int(rng.integers(0, 31))
        p_th = float(rng.uniform(0.50, 0.95))
    elif focus == "neg":
        z_entry, z_exit = _sample_z_pair(
            rng,
            z_entry_min=0.55,
            z_entry_max=1.25,
            z_exit_min=0.20,
            z_exit_max=1.00,
            min_gap=min_gap,
        )
        long_only = _sample_bool(rng, 0.35)
        short_filter = _sample_bool(rng, 0.20)
        gate_enforce = _sample_bool(rng, 0.30)
        max_hold = int(rng.integers(3, 26))
        cooldown = int(rng.integers(0, 6))
        p_th = float(rng.uniform(0.50, 0.65))
    elif focus == "pos":
        z_entry, z_exit = _sample_z_pair(
            rng,
            z_entry_min=0.80,
            z_entry_max=2.20,
            z_exit_min=0.20,
            z_exit_max=1.20,
            min_gap=min_gap,
        )
        long_only = _sample_bool(rng, 0.60)
        short_filter = _sample_bool(rng, 0.60)
        gate_enforce = _sample_bool(rng, 0.75)
        max_hold = int(rng.integers(6, 31))
        cooldown = int(rng.integers(2, 16))
        p_th = float(rng.uniform(0.65, 0.95))
    else:
        raise ValueError(f"Unsupported focus bucket: {focus}")
    return Candidate(
        focus=focus,
        long_only=bool(long_only),
        short_filter=bool(short_filter),
        gate_enforce=bool(gate_enforce),
        z_entry=float(z_entry),
        z_exit=float(z_exit),
        max_hold=int(max_hold),
        cooldown=int(cooldown),
        p_th=float(p_th),
    )


def _build_sampling_plan(n_runs: int) -> List[str]:
    n = max(int(n_runs), 1)
    n_uniform = int(n * 0.60)
    n_neg = int(n * 0.25)
    n_pos = n - n_uniform - n_neg
    return (["uniform"] * n_uniform) + (["neg"] * n_neg) + (["pos"] * n_pos)


def _build_base_config(
    input_file: Optional[str],
    pair: str,
    seed: int,
    signal_mode: str,
) -> PlaybookConfig:
    base_cfg = playbook_vecm.parse_args([])
    payload = base_cfg.to_dict()
    payload.update(
        {
            "input_file": input_file or base_cfg.input_file,
            "subset": pair,
            "method": "TVECM",
            "seed": int(seed),
            "signal_mode": str(signal_mode).lower(),
        }
    )
    return PlaybookConfig(**payload)


def _prepare_feature_cache(
    pair: str,
    base_cfg: PlaybookConfig,
    combos: Iterable[Tuple[bool, bool, bool]],
    scan_id: str,
) -> Mapping[Tuple[bool, bool, bool], FeatureBuildResult]:
    data_frame = playbook_vecm.load_and_validate_data(base_cfg.input_file)
    feature_cache: Dict[Tuple[bool, bool, bool], FeatureBuildResult] = {}
    for long_only, short_filter, gate_enforce in combos:
        combo_cfg = dataclasses.replace(
            base_cfg,
            subset=pair,
            long_only=bool(long_only),
            short_filter=bool(short_filter),
            gate_enforce=bool(gate_enforce),
        )
        combo_id = f"lo{int(long_only)}_sf{int(short_filter)}_ge{int(gate_enforce)}"
        feature_result = playbook_vecm.build_features(
            pair,
            playbook_vecm.FeatureConfig(
                base_config=combo_cfg,
                pair=pair,
                method=combo_cfg.method,
                horizon=combo_cfg.horizon,
                data_frame=data_frame,
                run_id=f"{scan_id}_{combo_id}",
            ),
        )
        feature_cache[(bool(long_only), bool(short_filter), bool(gate_enforce))] = feature_result
    return feature_cache


def _evaluate_candidate(
    pair: str,
    candidate: Candidate,
    feature_result: FeatureBuildResult,
    *,
    ann_days: int,
    scan_id: str,
    index: int,
) -> Dict[str, object]:
    result = playbook_vecm.evaluate_rules(
        feature_result,
        DecisionParams(
            z_entry=float(candidate.z_entry),
            z_exit=float(candidate.z_exit),
            max_hold=int(candidate.max_hold),
            cooldown=int(candidate.cooldown),
            p_th=float(candidate.p_th),
            run_id=f"{scan_id}_{index:04d}",
        ),
    )
    metrics = result.get("metrics", {})
    execution = result.get("execution")
    sharpe_oos = float(metrics.get("sharpe_oos", 0.0))
    maxdd = float(metrics.get("maxdd", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    turnover_annualised = float(metrics.get("turnover_annualised", 0.0))
    n_trades = int(metrics.get("n_trades", 0))
    sharpe_full = 0.0
    if execution is not None and hasattr(execution, "ret"):
        sharpe_full = compute_sharpe_full(getattr(execution, "ret"), ann_days)
        trades_frame = getattr(execution, "trades", None)
        if isinstance(trades_frame, pd.DataFrame):
            n_trades = int(trades_frame.shape[0])
    return {
        "run_id": str(result.get("run_id", f"{scan_id}_{index:04d}")),
        "pair": pair,
        "focus": candidate.focus,
        "sharpe_full": float(sharpe_full),
        "sharpe_oos": float(sharpe_oos),
        "n_trades": int(n_trades),
        "cagr": float(cagr),
        "maxdd": float(maxdd),
        "turnover_annualised": float(turnover_annualised),
        "long_only": bool(candidate.long_only),
        "short_filter": bool(candidate.short_filter),
        "gate_enforce": bool(candidate.gate_enforce),
        "z_entry": float(candidate.z_entry),
        "z_exit": float(candidate.z_exit),
        "max_hold": int(candidate.max_hold),
        "cooldown": int(candidate.cooldown),
        "p_th": float(candidate.p_th),
        "error": "",
    }


def _build_prefix_path(out_prefix: str) -> Path:
    candidate = Path(out_prefix)
    if candidate.parent == Path("."):
        return OUT_DIR / candidate.name
    return candidate


def run_extreme_scan(
    *,
    pair: str,
    n_runs: int,
    mode: str,
    signal_mode: str,
    min_trades: int,
    top_k: int,
    seed: int,
    out_prefix: str,
    input_file: Optional[str],
) -> ScanArtifacts:
    if mode != MODE_FULL_SAMPLE:
        raise ValueError(f"Unsupported mode: {mode}")
    normalized_pair = _normalize_pair(pair)
    rng = np.random.default_rng(int(seed))
    plan = _build_sampling_plan(int(n_runs))
    candidates = [_sample_candidate(rng, focus) for focus in plan]
    combos = sorted({(c.long_only, c.short_filter, c.gate_enforce) for c in candidates})
    base_cfg = _build_base_config(input_file, normalized_pair, seed=seed, signal_mode=signal_mode)
    scan_id = f"extreme_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    feature_cache = _prepare_feature_cache(normalized_pair, base_cfg, combos, scan_id)
    rows: List[Dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        combo = (candidate.long_only, candidate.short_filter, candidate.gate_enforce)
        feature_result = feature_cache[combo]
        try:
            row = _evaluate_candidate(
                normalized_pair,
                candidate,
                feature_result,
                ann_days=base_cfg.ann_days if base_cfg.ann_days else 252,
                scan_id=scan_id,
                index=idx,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            row = {
                "run_id": f"{scan_id}_{idx:04d}",
                "pair": normalized_pair,
                "focus": candidate.focus,
                "sharpe_full": float("nan"),
                "sharpe_oos": float("nan"),
                "n_trades": 0,
                "cagr": float("nan"),
                "maxdd": float("nan"),
                "turnover_annualised": float("nan"),
                "long_only": bool(candidate.long_only),
                "short_filter": bool(candidate.short_filter),
                "gate_enforce": bool(candidate.gate_enforce),
                "z_entry": float(candidate.z_entry),
                "z_exit": float(candidate.z_exit),
                "max_hold": int(candidate.max_hold),
                "cooldown": int(candidate.cooldown),
                "p_th": float(candidate.p_th),
                "error": str(exc),
            }
        rows.append(row)

    candidates_df = pd.DataFrame(rows)
    valid_df = filter_valid_candidates(candidates_df, min_trades=min_trades)
    top_pos, top_neg = rank_extremes(valid_df, top_k=top_k)

    quantiles = valid_df["sharpe_full"].quantile([0.05, 0.5, 0.95]) if not valid_df.empty else pd.Series(dtype=float)
    summary = {
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "pair": normalized_pair,
        "mode": mode,
        "signal_mode": str(signal_mode).lower(),
        "seed": int(seed),
        "n_runs": int(len(candidates_df)),
        "valid_runs": int(len(valid_df)),
        "min_trades": int(min_trades),
        "top_k": int(top_k),
        "distribution": {
            "median_sharpe_full": float(valid_df["sharpe_full"].median()) if not valid_df.empty else float("nan"),
            "q05_sharpe_full": float(quantiles.get(0.05, float("nan"))),
            "q95_sharpe_full": float(quantiles.get(0.95, float("nan"))),
            "valid_ratio": float(len(valid_df) / len(candidates_df)) if len(candidates_df) else 0.0,
        },
        "top_positive": top_pos.to_dict(orient="records"),
        "top_negative": top_neg.to_dict(orient="records"),
    }

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = _build_prefix_path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    candidates_path = prefix.parent / f"{prefix.name}_candidates_{timestamp}.csv"
    summary_path = prefix.parent / f"{prefix.name}_top_{timestamp}.json"
    candidates_df.to_csv(candidates_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return ScanArtifacts(
        candidates_path=candidates_path,
        summary_path=summary_path,
        total_runs=int(len(candidates_df)),
        valid_runs=int(len(valid_df)),
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search extreme full-sample Sharpe for a pair.")
    parser.add_argument("--pair", default="ANTM.JK,TLKM.JK", help="Pair in format AAA,BBB")
    parser.add_argument("--input", dest="input_file", default=None, help="Path to adj_close_data.csv")
    parser.add_argument("--n-runs", type=int, default=1000, help="Total sampled runs")
    parser.add_argument("--mode", default=MODE_FULL_SAMPLE, choices=[MODE_FULL_SAMPLE], help="Objective mode")
    parser.add_argument(
        "--signal-mode",
        default="normal",
        choices=["normal", "long_from_short_only"],
        help="Signal mode passed to playbook execution",
    )
    parser.add_argument("--min-trades", type=int, default=10, help="Minimum trades for valid ranking")
    parser.add_argument("--top-k", type=int, default=10, help="Rows for top positive and negative sets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out-prefix",
        default="extreme_sharpe",
        help="Output prefix (name or path prefix)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    artifacts = run_extreme_scan(
        pair=args.pair,
        n_runs=args.n_runs,
        mode=args.mode,
        signal_mode=args.signal_mode,
        min_trades=args.min_trades,
        top_k=args.top_k,
        seed=args.seed,
        out_prefix=args.out_prefix,
        input_file=args.input_file,
    )
    print(
        "Extreme Sharpe scan complete: "
        f"runs={artifacts.total_runs} valid={artifacts.valid_runs} "
        f"candidates_csv={artifacts.candidates_path} summary_json={artifacts.summary_path}"
    )


if __name__ == "__main__":
    main()
