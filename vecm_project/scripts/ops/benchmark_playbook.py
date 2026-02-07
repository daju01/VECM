"""Lightweight performance benchmark + profiler for the VECM playbook."""
from __future__ import annotations

import argparse
import cProfile
import time
from pathlib import Path
from typing import Optional

from vecm_project.scripts import playbook_vecm
from vecm_project.scripts.playbook_types import PlaybookConfig


def _build_config(input_file: str, subset: str, *, stage: int, seed: Optional[int]) -> PlaybookConfig:
    """Construct a PlaybookConfig for benchmark runs."""
    base_cfg = playbook_vecm.parse_args([])
    payload = base_cfg.to_dict()
    payload.update(
        {
            "input_file": input_file,
            "subset": subset,
            "stage": stage,
        }
    )
    if seed is not None:
        payload["seed"] = seed
    return PlaybookConfig(**payload)


def _run_once(cfg: PlaybookConfig, *, persist: bool) -> float:
    """Execute the playbook once and return the elapsed seconds."""
    data_frame = playbook_vecm.load_and_validate_data(cfg.input_file)
    start = time.perf_counter()
    playbook_vecm.run_playbook(cfg, persist=persist, data_frame=data_frame)
    return time.perf_counter() - start


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for benchmark runs."""
    parser = argparse.ArgumentParser(description="Benchmark VECM playbook runtime.")
    parser.add_argument("--input", dest="input_file", help="Path to adj_close_data.csv")
    parser.add_argument("--subset", default="ANTM,INCO", help="Pair subset to benchmark")
    parser.add_argument("--stage", type=int, default=1, help="Pipeline stage to execute")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--iters", type=int, default=1, help="Number of benchmark iterations")
    parser.add_argument("--persist", action="store_true", help="Persist artifacts while benchmarking")
    parser.add_argument("--profile", type=Path, help="Write cProfile stats to this path")
    args = parser.parse_args(argv)

    input_file = args.input_file or playbook_vecm.parse_args([]).input_file
    cfg = _build_config(input_file, args.subset, stage=args.stage, seed=args.seed)

    durations = []
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        durations.append(_run_once(cfg, persist=args.persist))
        profiler.disable()
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.profile))
    else:
        for _ in range(max(1, args.iters)):
            durations.append(_run_once(cfg, persist=args.persist))

    avg = sum(durations) / len(durations)
    print(f"Benchmark completed: iters={len(durations)} avg_sec={avg:.2f} last_sec={durations[-1]:.2f}")


if __name__ == "__main__":
    main()
