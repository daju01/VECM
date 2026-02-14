"""Stage-2 Bayesian optimisation mirroring the ParBayesianOptimization workflow."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import optuna
import pandas as pd

from . import storage
from .playbook_vecm import (
    DecisionParams,
    FeatureConfig,
    PlaybookConfig,
    build_features,
    evaluate_rules,
    load_and_validate_data,
    parse_args,
)

LOGGER = storage.configure_logging("stage2_bo")


@dataclass(frozen=True)
class Bound:
    """Simple bound container for optimisation variables."""

    low: float
    high: float
    is_int: bool = False


BO_BOUNDS: Dict[str, Bound] = {
    "z_entry": Bound(0.5, 2.0),
    "z_exit": Bound(0.2, 1.0),
    "max_hold": Bound(2, 15, is_int=True),
    "cooldown": Bound(0, 10, is_int=True),
    "p_th": Bound(0.50, 0.95),
}


def _base_config_dict(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    default_cfg = parse_args([])
    base: Dict[str, Any] = default_cfg.to_dict()
    if cfg:
        base.update({k: v for k, v in dict(cfg).items() if v is not None})
    return base


def _suggest_params(trial: optuna.Trial) -> Dict[str, float]:
    """Sample parameters for a given Optuna trial."""

    params: Dict[str, float] = {}
    for name, bound in BO_BOUNDS.items():
        if bound.is_int:
            params[name] = float(trial.suggest_int(name, int(bound.low), int(bound.high)))
        else:
            params[name] = float(trial.suggest_float(name, bound.low, bound.high))
    return params


def _resolve_input_file(base_cfg: Mapping[str, Any]) -> str:
    path = str(base_cfg.get("input_file") or "")
    if path:
        return path
    default_cfg = parse_args([])
    return default_cfg.input_file


def _resolve_turnover_lambda() -> float:
    lambda_str = os.getenv("STAGE2_LAMBDA_TURNOVER", "0.01")
    try:
        return float(lambda_str)
    except ValueError:
        return 0.01


def _resolve_min_trades() -> int:
    min_trades_str = os.getenv("STAGE2_MIN_TRADES", "5")
    try:
        min_trades = int(min_trades_str)
    except ValueError:
        min_trades = 5
    return max(0, min_trades)


def _resolve_huge_penalty() -> float:
    raw = os.getenv("STAGE2_HUGE_PENALTY", "1000000")
    try:
        return float(raw)
    except ValueError:
        return 1_000_000.0


def _resolve_obj_mode() -> str:
    mode = str(os.getenv("STAGE2_OBJ_MODE", "legacy")).strip().lower()
    if mode not in {"legacy", "idx_v1", "idx_v2_calmar"}:
        return "legacy"
    return mode


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _cfg_or_env_float(cfg: Mapping[str, Any], key: str, env_name: str, default: float) -> float:
    """Read float from cfg first; fall back to env/default when cfg is None/invalid."""

    value = cfg.get(key)
    if value is None:
        return _env_float(env_name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return _env_float(env_name, default)


def _objective_env_snapshot(base_cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(base_cfg or {})
    return {
        "objective_mode": _resolve_obj_mode(),
        "cost_model": str(cfg.get("cost_model", "simple")),
        "broker_buy_rate": _cfg_or_env_float(cfg, "broker_buy_rate", "IDX_BROKER_BUY_RATE", 0.0019),
        "broker_sell_rate": _cfg_or_env_float(cfg, "broker_sell_rate", "IDX_BROKER_SELL_RATE", 0.0029),
        "exchange_levy": _cfg_or_env_float(cfg, "exchange_levy", "IDX_LEVY_RATE", 0.0),
        "sell_tax": _cfg_or_env_float(cfg, "sell_tax", "IDX_SELL_TAX_RATE", 0.001),
        "spread_bps": _cfg_or_env_float(cfg, "spread_bps", "IDX_SPREAD_BPS", 20.0),
        "impact_model": str(cfg.get("impact_model", os.getenv("IDX_IMPACT_MODEL", "sqrt"))),
        "impact_k": _cfg_or_env_float(cfg, "impact_k", "IDX_IMPACT_K", 1.0),
        "adtv_win": _cfg_or_env_float(cfg, "adtv_win", "IDX_ADTV_WIN", 20.0),
        "sigma_win": _cfg_or_env_float(cfg, "sigma_win", "IDX_SIGMA_WIN", 20.0),
        "illiq_cap_mode_cfg": str(cfg.get("illiq_cap_mode", os.getenv("IDX_ILLIQ_CAP_MODE", "insample_p80"))),
        "illiq_cap_value_cfg": _cfg_or_env_float(cfg, "illiq_cap_value", "IDX_ILLIQ_CAP_VALUE", float("nan")),
        "dd_cap": _env_float("IDX_DD_CAP", 0.15),
        "dd_hard": _env_float("IDX_DD_HARD", 0.20),
        "rho_cap": _env_float("IDX_RHO_CAP", 0.02),
        "illiq_cap_mode": str(os.getenv("IDX_ILLIQ_CAP_MODE", "insample_p80")).strip().lower(),
        "illiq_cap_value": _env_float("IDX_ILLIQ_CAP_VALUE", float("nan")),
        "lambda_dd": _env_float("IDX_LAMBDA_DD", 4.0),
        "lambda_cap": _env_float("IDX_LAMBDA_CAP", 50.0),
        "lambda_illiq": _env_float("IDX_LAMBDA_ILLIQ", 10.0),
        "lambda_to": _env_float("IDX_LAMBDA_TO", 0.002),
        "calmar_eps": _env_float("IDX_CALMAR_EPS", 0.01),
        "min_trades": _resolve_min_trades(),
        "huge_penalty": _resolve_huge_penalty(),
    }


def score_rules(
    *,
    feature_result: Any,
    decision_params: DecisionParams,
) -> Dict[str, Any]:
    """Evaluate the cached features once and expose scalar diagnostics."""

    start = time.perf_counter()
    result = evaluate_rules(feature_result, decision_params)
    elapsed = time.perf_counter() - start
    metrics = result.get("metrics", {})
    execution = result.get("execution")
    objective_mode = _resolve_obj_mode()
    sharpe = float(metrics.get("sharpe_oos", 0.0))
    sharpe_net = float(metrics.get("sharpe_oos_net", sharpe))
    turnover = float(metrics.get("turnover", 0.0))
    turnover_ann = float(metrics.get("turnover_annualised", turnover))
    turnover_ann_notional = float(metrics.get("turnover_annualised_notional", turnover_ann))
    n_trades = int(metrics.get("n_trades", 0) or 0)
    min_trades = _resolve_min_trades()
    huge_penalty = _resolve_huge_penalty()

    penalty_dd = 0.0
    penalty_cap = 0.0
    penalty_illiq = 0.0
    penalty_to = 0.0
    penalty_short = 0.0
    base_value = sharpe
    lambda_turnover = _resolve_turnover_lambda()
    score = sharpe - lambda_turnover * turnover_ann

    if objective_mode in {"idx_v1", "idx_v2_calmar"}:
        maxdd_net = float(metrics.get("maxdd_oos_net", abs(float(metrics.get("maxdd", 0.0)))))
        if maxdd_net < 0:
            maxdd_net = abs(maxdd_net)
        participation_mean = float(metrics.get("participation_mean", 0.0))
        amihud_illiq = float(metrics.get("amihud_illiq", 0.0))
        illiq_cap = float(metrics.get("illiq_cap", math.nan))
        if not math.isfinite(illiq_cap):
            illiq_cap = amihud_illiq

        rho_cap = _env_float("IDX_RHO_CAP", 0.02)
        lambda_cap = _env_float("IDX_LAMBDA_CAP", 50.0)
        lambda_illiq = _env_float("IDX_LAMBDA_ILLIQ", 10.0)
        lambda_to = _env_float("IDX_LAMBDA_TO", 0.002)
        lambda_dd = _env_float("IDX_LAMBDA_DD", 4.0)

        if objective_mode == "idx_v1":
            dd_cap = _env_float("IDX_DD_CAP", 0.15)
            base_value = sharpe_net
            penalty_dd = lambda_dd * max(0.0, maxdd_net - dd_cap) ** 2
        else:
            dd_hard = _env_float("IDX_DD_HARD", 0.20)
            cagr_net = float(metrics.get("cagr_oos_net", metrics.get("cagr", 0.0)))
            calmar_net = float(metrics.get("calmar_oos_net", 0.0))
            base_value = cagr_net if cagr_net <= 0 else calmar_net
            penalty_dd = lambda_dd * max(0.0, maxdd_net - dd_hard) ** 2

        penalty_cap = lambda_cap * max(0.0, participation_mean - rho_cap) ** 2
        penalty_illiq = lambda_illiq * max(0.0, amihud_illiq - illiq_cap) ** 2
        penalty_to = lambda_to * turnover_ann_notional
        score = base_value - penalty_dd - penalty_cap - penalty_illiq - penalty_to

        allow_short = True
        if getattr(feature_result, "features", None) is not None:
            allow_short = not bool(getattr(feature_result.features.cfg, "long_only", False))
        if not allow_short and execution is not None and hasattr(execution, "pos"):
            pos_series = getattr(execution, "pos")
            if isinstance(pos_series, pd.Series):
                oos_start = getattr(getattr(feature_result, "features", None), "oos_start_date", None)
                if oos_start is not None and hasattr(pos_series.index, "date"):
                    pos_oos = pos_series[pos_series.index.date >= oos_start]
                    if pos_oos.empty:
                        pos_oos = pos_series
                else:
                    pos_oos = pos_series
                if (pos_oos < 0).any():
                    penalty_short = huge_penalty
                    score -= penalty_short

    min_trades_penalty = 0.0
    valid_trial = n_trades >= min_trades
    if not valid_trial:
        min_trades_penalty = huge_penalty + float(min_trades - n_trades)
        score -= min_trades_penalty

    maxdd_metric = float(metrics.get("maxdd_oos_net", abs(float(metrics.get("maxdd", 0.0)))))
    if maxdd_metric < 0:
        maxdd_metric = abs(maxdd_metric)

    diagnostics = {
        "Score": float(score),
        "eval_time_s": float(elapsed),
        "objective_mode": float({"legacy": 0, "idx_v1": 1, "idx_v2_calmar": 2}.get(objective_mode, 0)),
        "objective_mode_label": objective_mode,
        "base_value": float(base_value),
        "sharpe_oos": sharpe,
        "sharpe_oos_net": sharpe_net,
        "maxdd": float(metrics.get("maxdd", 0.0)),
        "maxdd_oos_net": float(maxdd_metric),
        "turnover": turnover,
        "turnover_annualised": turnover_ann,
        "turnover_annualised_notional": turnover_ann_notional,
        "participation_mean": float(metrics.get("participation_mean", 0.0)),
        "participation_max": float(metrics.get("participation_max", 0.0)),
        "amihud_illiq": float(metrics.get("amihud_illiq", 0.0)),
        "illiq_cap": float(metrics.get("illiq_cap", math.nan)),
        "illiq_cap_used": float(metrics.get("illiq_cap_used", metrics.get("illiq_cap", math.nan))),
        "cost_model": str(metrics.get("cost_model", "simple")),
        "cagr_oos_net": float(metrics.get("cagr_oos_net", metrics.get("cagr", 0.0))),
        "calmar_oos_net": float(metrics.get("calmar_oos_net", 0.0)),
        "penalty_dd": float(penalty_dd),
        "penalty_cap": float(penalty_cap),
        "penalty_illiq": float(penalty_illiq),
        "penalty_to": float(penalty_to),
        "penalty_short": float(penalty_short),
        "n_trades": float(n_trades),
        "min_trades_required": float(min_trades),
        "min_trades_penalty": float(min_trades_penalty),
        "huge_penalty": float(huge_penalty),
        "valid_trial": float(1.0 if valid_trial else 0.0),
        "alpha_ec": float(metrics.get("alpha_ec", math.nan)),
        "half_life_full": float(metrics.get("half_life_full", math.nan)),
        "idx_dd_cap": float(_env_float("IDX_DD_CAP", 0.15)),
        "idx_dd_hard": float(_env_float("IDX_DD_HARD", 0.20)),
        "idx_rho_cap": float(_env_float("IDX_RHO_CAP", 0.02)),
        "idx_lambda_dd": float(_env_float("IDX_LAMBDA_DD", 4.0)),
        "idx_lambda_cap": float(_env_float("IDX_LAMBDA_CAP", 50.0)),
        "idx_lambda_illiq": float(_env_float("IDX_LAMBDA_ILLIQ", 10.0)),
        "idx_lambda_to": float(_env_float("IDX_LAMBDA_TO", 0.002)),
        "idx_calmar_eps": float(_env_float("IDX_CALMAR_EPS", 0.01)),
        "idx_spread_bps": float(metrics.get("idx_spread_bps", math.nan)),
        "idx_impact_k": float(metrics.get("idx_impact_k", math.nan)),
        "idx_sell_tax_rate": float(metrics.get("idx_sell_tax_rate", math.nan)),
        "idx_adtv_win": float(metrics.get("idx_adtv_win", math.nan)),
        "idx_sigma_win": float(metrics.get("idx_sigma_win", math.nan)),
        "illiq_cap_mode": str(metrics.get("illiq_cap_mode", "insample_p80")),
    }
    return diagnostics


def run_bo(
    *,
    pair: str,
    method: str = "TVECM",
    horizon: str = "oos_full",
    cfg: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    n_init: int = 4,
    iters: int = 12,
    acq: str = "ei",
    n_jobs: Optional[int] = None,
) -> optuna.Study:
    """Launch Bayesian optimisation with Optuna using the shared parameter grid."""

    quick_mode = os.getenv("VECM_BO_QUICK_MODE", "").lower() in {"1", "true", "yes", "on"}
    if quick_mode:
        n_init = 2
        iters = 4
        LOGGER.info("Quick BO mode active: VECM_BO_QUICK_MODE=1 (n_init=%d iters=%d)", n_init, iters)

    if n_init < 1:
        raise ValueError("n_init must be >= 1")
    if iters < 1:
        raise ValueError("iters must be >= 1")

    total_trials = n_init + iters
    study_run_id = run_id or f"stage2_bo_{pair.replace(',', '-')}_{int(time.time())}"
    cpu_default = max(1, min(4, (os.cpu_count() or 2) - 1))
    n_jobs = n_jobs or cpu_default
    LOGGER.info(
        "Stage2 BO start | pair=%s method=%s horizon=%s run_id=%s trials=%d n_jobs=%d",
        pair,
        method,
        horizon,
        study_run_id,
        total_trials,
        n_jobs,
    )
    start_wall = dt.datetime.utcnow()

    base_cfg = _base_config_dict(cfg)
    input_path = _resolve_input_file(base_cfg)
    data_frame = load_and_validate_data(input_path)
    cfg_payload: Dict[str, Any] = dict(base_cfg)
    cfg_payload.update(
        {
            "subset": pair,
            "method": method,
            "stage": 2,
            "notes": f"stage2_bo|pair={pair}|method={method}",
        }
    )
    if horizon:
        cfg_payload.setdefault("horizon", horizon)
    base_playbook_cfg = PlaybookConfig(**cfg_payload)
    feature_result = build_features(
        pair,
        FeatureConfig(
            base_config=base_playbook_cfg,
            pair=pair,
            method=method,
            horizon=horizon,
            data_frame=data_frame,
            run_id=study_run_id,
        ),
    )
    if feature_result.skip_result is not None:
        LOGGER.warning(
            "Feature build returned a skipped result; BO trials will only evaluate cached output."
        )
    else:
        LOGGER.info("Feature cache ready; BO trials will only evaluate rules.")
    sampler_seed = base_cfg.get("seed") if isinstance(base_cfg, Mapping) else None
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=n_init)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        trial_run_id = f"{study_run_id}_trial{trial.number:03d}"
        decision_params = DecisionParams(
            z_entry=float(params["z_entry"]),
            z_exit=float(params["z_exit"]),
            max_hold=int(params["max_hold"]),
            cooldown=int(params["cooldown"]),
            p_th=float(params["p_th"]),
            run_id=trial_run_id,
        )
        diagnostics = score_rules(
            feature_result=feature_result,
            decision_params=decision_params,
        )
        record = {
            "params": params,
            "diagnostics": diagnostics,
            "trial_run_id": trial_run_id,
        }
        trial.set_user_attr("record", record)
        LOGGER.info(
            "BO trial=%s Score=%.4f sharpe=%.4f maxdd=%.4f turnover=%.4f t_ann=%.4f trades=%d p_th=%.3f valid=%s",
            trial.number,
            diagnostics["Score"],
            diagnostics["sharpe_oos"],
            diagnostics["maxdd"],
            diagnostics["turnover"],
            diagnostics["turnover_annualised"],
            int(diagnostics["n_trades"]),
            float(params["p_th"]),
            bool(int(diagnostics["valid_trial"])),
        )
        return diagnostics["Score"]

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=total_trials, n_jobs=n_jobs, show_progress_bar=False)

    finish_wall = dt.datetime.utcnow()
    best = study.best_trial
    LOGGER.info("Stage2 BO complete | best_trial=%s best_score=%.4f", best.number, best.value)

    with storage.managed_storage("stage2_bo") as conn:
        with storage.with_transaction(conn):
            objective_snapshot = _objective_env_snapshot(cfg_payload)
            best_record = best.user_attrs.get("record") if hasattr(best, "user_attrs") else None
            if isinstance(best_record, dict):
                best_diag = best_record.get("diagnostics", {})
                if isinstance(best_diag, dict):
                    objective_snapshot["illiq_cap_used_best"] = best_diag.get("illiq_cap_used")
                    objective_snapshot["objective_mode_label"] = best_diag.get("objective_mode_label")
            storage.write_run(
                conn,
                study_run_id,
                started_at=start_wall,
                finished_at=finish_wall,
                n_workers=n_jobs,
                plan="optuna_tpe",
                seed_method="tpe",
                notes=json.dumps({"pair": pair, "method": method, "acq": acq, **objective_snapshot}),
            )
            horizon_payload = {"horizon": horizon} if horizon else None
            for trial in study.trials:
                record = trial.user_attrs.get("record")
                if not record:
                    continue
                trial_id = f"{pair.replace(',', '-')}:{trial.number:03d}"
                params_payload = {
                    **record["params"],
                    "trial_run_id": record["trial_run_id"],
                }
                diagnostics = record["diagnostics"]
                storage.write_trial(
                    conn,
                    run_id=study_run_id,
                    trial_id=trial_id,
                    stage=2,
                    pair=pair,
                    method=method,
                    params=params_payload,
                    horizon=horizon_payload,
                    eval_time_s=diagnostics["eval_time_s"],
                    sharpe_oos=diagnostics["sharpe_oos"],
                    maxdd=diagnostics["maxdd"],
                    turnover=diagnostics["turnover"],
                    alpha_ec=diagnostics.get("alpha_ec"),
                    half_life_full=diagnostics.get("half_life_full"),
                    pruned=False,
                )
            storage.mark_run_finished(conn, study_run_id, finished_at=finish_wall)

    return study


if __name__ == "__main__":
    try:
        study = run_bo(pair="ANTM,INCO", n_init=1, iters=1)
        print("Best trial:", study.best_trial.number)
    except FileNotFoundError as exc:
        print(exc)
