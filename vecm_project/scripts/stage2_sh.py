"""Stage-2 Successive Halving / Hyperband driver mirroring the R workflow."""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import optuna
import pandas as pd

from . import storage
from .playbook_vecm import (
    PlaybookConfig,
    load_and_validate_data,
    parse_args,
    run_playbook,
)

LOGGER = storage.configure_logging("stage2_sh")


@dataclass(frozen=True)
class Bound:
    """Simple container describing the optimisation bounds."""

    low: float
    high: float
    is_int: bool = False


SH_BOUNDS: Dict[str, Bound] = {
    "z_entry": Bound(0.5, 2.0),
    "z_exit": Bound(0.2, 1.0),
    "max_hold": Bound(2, 15, is_int=True),
    "cooldown": Bound(0, 10, is_int=True),
}

DEFAULT_HORIZONS: Tuple[str, ...] = ("short", "long")


def _base_config_dict(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    default_cfg = parse_args([])
    base: Dict[str, Any] = default_cfg.to_dict()
    if cfg:
        base.update({k: v for k, v in dict(cfg).items() if v is not None})
    return base


def _resolve_input_file(base_cfg: Mapping[str, Any]) -> str:
    path = str(base_cfg.get("input_file") or "")
    if path:
        return path
    default_cfg = parse_args([])
    return default_cfg.input_file


def _suggest_params(trial: optuna.Trial) -> Dict[str, float]:
    """Sample optimisation parameters following the shared bounds."""

    params: Dict[str, float] = {}
    for name, bound in SH_BOUNDS.items():
        if bound.is_int:
            params[name] = float(trial.suggest_int(name, int(bound.low), int(bound.high)))
        else:
            params[name] = float(trial.suggest_float(name, bound.low, bound.high))
    return params


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
    step: int,
) -> Dict[str, float]:
    """Execute the playbook once and expose diagnostics for the optimiser."""

    cfg_payload: Dict[str, Any] = dict(base_cfg)
    cfg_payload.update(
        {
            "subset": pair,
            "method": method,
            "stage": 2,
            "notes": f"stage2_sh|pair={pair}|method={method}|horizon={horizon}|step={step}",
            "max_hold": int(max_hold),
            "cooldown": int(cooldown),
            "z_exit": float(z_exit),
            # Use z_entry as a guard for the stop/entry threshold similar to the R code.
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
    diagnostics = {
        "loss": float(-metrics.get("sharpe_oos", 0.0)),
        "eval_time_s": float(elapsed),
        "sharpe_oos": float(metrics.get("sharpe_oos", 0.0)),
        "maxdd": float(metrics.get("maxdd", 0.0)),
        "turnover": float(metrics.get("turnover", 0.0)),
    }
    return diagnostics


def run_successive_halving(
    *,
    pair: str,
    method: str = "TVECM",
    cfg: Optional[Mapping[str, Any]] = None,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    eta: int = 2,
    n_trials: int = 12,
    run_id: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> optuna.Study:
    """Launch a successive halving search mirroring the R reference implementation."""

    if not horizons:
        raise ValueError("At least one horizon must be provided")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    study_run_id = run_id or f"stage2_sh_{pair.replace(',', '-')}_{int(time.time())}"
    cpu_default = max(1, min(4, (os.cpu_count() or 2) - 1))
    n_jobs = n_jobs or cpu_default
    LOGGER.info(
        "Stage2 SH start | pair=%s method=%s run_id=%s horizons=%s trials=%d n_jobs=%d",
        pair,
        method,
        study_run_id,
        list(horizons),
        n_trials,
        n_jobs,
    )

    base_cfg = _base_config_dict(cfg)
    input_path = _resolve_input_file(base_cfg)
    data_frame = load_and_validate_data(input_path)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=eta)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    start_wall = dt.datetime.utcnow()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        trial.set_user_attr("params", params)
        rung_records: List[Dict[str, Any]] = []

        for step, horizon in enumerate(horizons, start=1):
            rung_run_id = f"{study_run_id}_trial{trial.number:03d}_step{step}"
            diagnostics = score_playbook(
                pair=pair,
                method=method,
                horizon=horizon,
                base_cfg=base_cfg,
                data_frame=data_frame,
                step=step,
                **params,
            )
            loss = diagnostics["loss"]
            rung_records.append(
                {
                    "step": step,
                    "horizon": horizon,
                    "diagnostics": diagnostics,
                    "trial_run_id": rung_run_id,
                }
            )
            trial.set_user_attr("records", rung_records)
            trial.report(loss, step=step)
            LOGGER.info(
                "SH trial=%s step=%s horizon=%s sharpe=%.4f loss=%.4f",
                trial.number,
                step,
                horizon,
                diagnostics["sharpe_oos"],
                loss,
            )
            if trial.should_prune():
                trial.set_user_attr("pruned", True)
                raise optuna.TrialPruned()

        trial.set_user_attr("pruned", False)
        return rung_records[-1]["diagnostics"]["loss"]

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    finish_wall = dt.datetime.utcnow()
    best = study.best_trial
    best_sharpe = -best.value
    LOGGER.info(
        "Stage2 SH complete | best_trial=%s best_loss=%.4f best_sharpe=%.4f",
        best.number,
        best.value,
        best_sharpe,
    )

    with storage.managed_storage("stage2_sh") as conn:
        storage.write_run(
            conn,
            study_run_id,
            started_at=start_wall,
            finished_at=finish_wall,
            n_workers=n_jobs,
            plan="optuna_hyperband",
            seed_method="hyperband",
            notes=json.dumps({"pair": pair, "method": method, "eta": eta}),
        )

        for trial in study.trials:
            params = trial.user_attrs.get("params", {})
            rung_records = trial.user_attrs.get("records", [])
            is_pruned = trial.user_attrs.get(
                "pruned", trial.state == optuna.trial.TrialState.PRUNED
            )
            for record in rung_records:
                diagnostics = record["diagnostics"]
                horizon = record["horizon"]
                step = record["step"]
                trial_id = (
                    f"{pair.replace(',', '-')}" f":sh:{trial.number:03d}:step{step}"
                )
                params_payload = {
                    **params,
                    "trial_run_id": record["trial_run_id"],
                    "step": step,
                }
                storage.write_trial(
                    conn,
                    run_id=study_run_id,
                    trial_id=trial_id,
                    stage=2,
                    pair=pair,
                    method=method,
                    params=params_payload,
                    horizon={"horizon": horizon},
                    eval_time_s=diagnostics["eval_time_s"],
                    sharpe_oos=diagnostics["sharpe_oos"],
                    maxdd=diagnostics["maxdd"],
                    turnover=diagnostics["turnover"],
                    pruned=bool(is_pruned and step == rung_records[-1]["step"]),
                )

        storage.mark_run_finished(conn, study_run_id, finished_at=finish_wall)

    return study


if __name__ == "__main__":
    try:
        study = run_successive_halving(pair="ANTM,INCO", n_trials=3, horizons=("short", "long"))
        print("Best SH trial:", study.best_trial.number)
    except FileNotFoundError as exc:
        print(exc)
