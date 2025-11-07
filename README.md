# VECM

for stock market price analysis

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r vecm_project/requirements.txt
# The loader will fetch adjusted close data from Yahoo Finance on first run
python vecm_project/run_demo.py
```

### When `pip install` is blocked

Some sandboxed or corporate environments intercept outbound HTTPS traffic and
return `403 Forbidden` for the Python Package Index. If you see repeated
`Cannot connect to proxy` errors while installing the requirements, try one of
the following approaches:

1. **Use WSL or any machine with open internet access.** Clone the repository
   inside your WSL distribution (or another Linux/macOS host) and run the
   commands above there. Once the virtual environment is populated you can run
   the demo directly from that machine.
2. **Pre-download the wheels.** On a machine with internet access execute
   `pip download -r vecm_project/requirements.txt -d wheels/`, copy the
   resulting `wheels/` directory to the restricted environment, and install via
   `pip install --no-index --find-links wheels -r vecm_project/requirements.txt`.
3. **Point pip at an internal mirror.** If your organisation provides a
   whitelisted PyPI mirror, export `PIP_INDEX_URL=<mirror-url>` before running
   `pip install`.

Without at least one of these workarounds the Python interpreter will be
missing dependencies such as `duckdb`, `pandas`, and `statsmodels`, and the
demo script will terminate immediately with `ModuleNotFoundError`.

## Testing

To run the lightweight sanity check that we ship with the project, make sure
the dependencies above are installed and then execute:

```bash
python -m compileall vecm_project
```

The command only compiles the sources, so it runs quickly and does not require
network access once the packages are available in your virtual environment.

## Runtime controls

The Python translation ships with conservative defaults so a full demo run
completes within roughly an hour on a laptop. You can tune the workload via
environment variables:

* ``VECM_MAX_GRID`` limits the number of jobs spawned by ``parallel_run``
  (default ``48``). Set a higher value when you have more time to sweep the
  Stage-1 grid.
* ``run_bo`` now launches 16 trials by default (4 initial points + 12 TPE
  steps) and caps parallel workers at four logical cores. Override ``n_init``
  or ``iters`` when you want a deeper Bayesian search.
* ``run_successive_halving`` evaluates up to 12 trials across the ``("short",
  "long")`` horizons and also restricts parallelism to four workers by
  default. Increase ``n_trials`` or pass a custom ``horizons`` tuple for a more
  exhaustive pass.

These defaults keep the optimisation passes responsive while still leaving room
for users to scale up the search when more compute time is available.
