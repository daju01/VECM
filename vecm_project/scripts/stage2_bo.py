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
    PlaybookConfig,
    load_and_validate_data,
    parse_args,
    run_playbook,
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


def score_playbook(
    *,
    pair: str,
    method: str,
    horizon: str,
    base_cfg: Mapping[str, Any],
    data_frame: pd.DataFrame,
    z_entry: float,
    z_exit: float,
    max_hold: float,
    cooldown: float,
) -> Dict[str, float]:
    """Execute the playbook once and expose scalar diagnostics."""

    cfg_payload: Dict[str, Any] = dict(base_cfg)
    cfg_payload.update(
        {
            "subset": pair,
            "method": method,
            "stage": 2,
            "notes": f"stage2_bo|pair={pair}|method={method}",
            "z_entry": float(z_entry),
            "max_hold": int(max_hold),
            "cooldown": int(cooldown),
            "z_exit": float(z_exit),
            "z_stop": float(max(z_entry, z_exit)),
        }
    )
    if horizon:
        cfg_payload.setdefault("horizon", horizon)

    cfg = PlaybookConfig(**cfg_payload)
    start = time.perf_counter()
    result = run_playbook(cfg, persist=False, data_frame=data_frame)
    elapsed = time.perf_counter() - start
    metrics = result.get("metrics", {})
    sharpe = float(metrics.get("sharpe_oos", 0.0))
    turnover = float(metrics.get("turnover", 0.0))
    turnover_ann = float(metrics.get("turnover_annualised", turnover))

    lambda_str = os.getenv("STAGE2_LAMBDA_TURNOVER", "0.01")
    try:
        lambda_turnover = float(lambda_str)
    except ValueError:
        lambda_turnover = 0.01

    score = sharpe - lambda_turnover * turnover_ann

    diagnostics = {
        "Score": float(score),
        "eval_time_s": float(elapsed),
        "sharpe_oos": sharpe,
        "maxdd": float(metrics.get("maxdd", 0.0)),
        "turnover": turnover,
        "turnover_annualised": turnover_ann,
        "alpha_ec": float(metrics.get("alpha_ec", math.nan)),
        "half_life_full": float(metrics.get("half_life_full", math.nan)),
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
    sampler_seed = base_cfg.get("seed") if isinstance(base_cfg, Mapping) else None
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=n_init)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        trial_run_id = f"{study_run_id}_trial{trial.number:03d}"
        diagnostics = score_playbook(
            pair=pair,
            method=method,
            horizon=horizon,
            base_cfg=base_cfg,
            data_frame=data_frame,
            **params,
        )
        record = {
            "params": params,
            "diagnostics": diagnostics,
            "trial_run_id": trial_run_id,
        }
        trial.set_user_attr("record", record)
        LOGGER.info(
            "BO trial=%s Score=%.4f sharpe=%.4f maxdd=%.4f turnover=%.4f t_ann=%.4f",
            trial.number,
            diagnostics["Score"],
            diagnostics["sharpe_oos"],
            diagnostics["maxdd"],
            diagnostics["turnover"],
            diagnostics["turnover_annualised"],
        )
        return diagnostics["Score"]

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=total_trials, n_jobs=n_jobs, show_progress_bar=False)

    finish_wall = dt.datetime.utcnow()
    best = study.best_trial
    LOGGER.info("Stage2 BO complete | best_trial=%s best_score=%.4f", best.number, best.value)

    with storage.managed_storage("stage2_bo") as conn:
        storage.write_run(
            conn,
            study_run_id,
            started_at=start_wall,
            finished_at=finish_wall,
            n_workers=n_jobs,
            plan="optuna_tpe",
            seed_method="tpe",
            notes=json.dumps({"pair": pair, "method": method, "acq": acq}),
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
