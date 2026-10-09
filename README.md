# AEFC-FRL Experimental Code

This repository contains the experimental code for AEFC-FRL security mechanisms
and a compact PyTorch federated reinforcement learning reference implementation.
It separates three evidence paths: a reduced-order CPS mechanism study, a native
IEEE 39-bus MATLAB/Simulink workflow, and a lightweight `Pendulum-v1` federated
learner used for implementation diagnostics.

The native workflow resolves the official MathWorks `IEEE39BusSystem` example at
runtime and validates the observation/actuation mapping before running the
experiment. The MathWorks model, generated mappings, raw traces, and `results/`
artifacts are not stored in this repository. Scripts containing embedded summary
arrays reproduce plots from those values; they do not regenerate the underlying
power-grid runs.

## Included Components

- Provenance-aware temporal belief updating, a risk-aware recovery gate,
  trust-gated adaptation, robust sparse aggregation, and a safety shield.
- A seven-method, four-scenario reduced-order CPS experiment with per-step JSONL
  traces, per-seed summaries, and a provenance manifest.
- MATLAB R2024b scripts for official IEEE 39-bus model discovery, causal interface
  validation, model instrumentation, and native matched-seed experiments.
- AEFC, FedAvg, and FedProx PyTorch aggregation paths with DDPG-style local agents.
- Random-update, poisoning, and reward-flip settings for the PyTorch reference
  learner.
- Round-level diagnostic exports for update norms, client heterogeneity,
  compression residuals, communication, and evaluation returns.
- A self-contained traffic-signal interface study with multi-seed CSV and JSON
  outputs.
- Standalone Matplotlib scripts for the manuscript figures.

## Requirements

- Python 3.9
- PyTorch 2.0.1 for the reference learner
- MATLAB and Simulink R2024b or later for the native IEEE 39-bus workflow
- CPU execution is supported; CUDA is used automatically when available.

Create an isolated environment and install the pinned dependencies.

### Linux or macOS

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reduced-Order CPS Experiment

Run the configured 20 seeds, seven methods, and four attack scenarios:

```bash
python experiments/run_joint.py \
  --config configs/joint20.json \
  --output results/joint20
python analysis/make_results.py
```

For a short smoke run, add `--seeds 1 --steps 20`. The experiment records raw
JSONL traces, seed-level summaries, `summary.csv`, and `manifest.json` beneath the
selected output directory. The fairness-calibrated mechanism comparison is run
separately:

```bash
python experiments/run_mechanism_calibration.py
```

The scope and fixed provenance rules for supplemental experiments are documented
in `experiments/SUPPLEMENTAL_PROTOCOL.md`.

## Native IEEE 39-Bus Workflow

Run MATLAB from the repository root with MATLAB and Simulink R2024b or later:

```matlab
addpath('matlab_r2024b', 'experiments/matlab')
aefc_ieee39_bootstrap
aefc_ieee39_interface_discovery
aefc_ieee39_mapping_probe
aefc_ieee39_build_native_model
aefc_ieee39_native_joint20
```

The bootstrap resolves the official MathWorks example, and the discovery and
probe stages must produce a validated mapping before model construction or native
evaluation can continue. After a complete native run, generate the checked
statistical summaries and figures with:

```bash
python analysis/make_ieee39_results.py
```

The analysis script refuses incomplete experiment matrices, mixed commit/config
provenance, stale mappings, or non-finite measurements.

## PyTorch Reference Learner

Run AEFC with ten clients, a 20% sparsification ratio, and 10% random malicious
clients:

```bash
python src/main.py \
  --algorithm AEFC \
  --env Pendulum-v1 \
  --num_agents 10 \
  --episodes 500 \
  --sync_interval 5 \
  --malicious_frac 0.1 \
  --attack_type random \
  --kappa 0.2 \
  --seed 42
```

On PowerShell, place the command on one line or replace each trailing `\` with a
backtick. Training logs are written to `runs/` by default.

To use the supplied configuration file:

```bash
python src/main.py --config configs/example_config.yaml
```

Command-line values that differ from parser defaults override the corresponding
YAML entries. The available algorithm names are `AEFC`, `FedAvg`, and `FedProx`;
the available attack names are `none`, `random`, `poison`, and `reward_flip`.

## Convergence Diagnostics

The diagnostic runner records round-level quantities over multiple seeds:

```bash
python scripts/run_diagnostics.py \
  --env Pendulum-v1 \
  --num_agents 5 \
  --episodes 40 \
  --sync_interval 5 \
  --kappa 0.2 \
  --seeds 42,43,44 \
  --out_dir diagnostics
```

It writes:

- `diagnostics/aefc_diagnostics_rounds.csv`
- `diagnostics/aefc_diagnostics_summary.json`

These diagnostics exercise the reference learner on `Pendulum-v1`; they are not
measurements from the omitted power-grid co-simulation.

## Traffic Proof of Concept

Run the self-contained traffic-signal study with:

```bash
python scripts/figures/Traffic-PoC.py
```

The script evaluates its configured methods over 15 seeds and writes
`traffic_poc_results.csv` and `traffic_poc_summary.json` beside the script.

## Figure Scripts

The remaining files under `scripts/figures/` are standalone plotting utilities.
Run a script from the repository root, for example:

```bash
python scripts/figures/Robustness.py
```

Matplotlib opens the generated figure in an interactive window. Some scripts use
embedded aggregate values retained for figure reproduction; inspect the script
header and data arrays before using a plot for a new experiment.

## Repository Layout

```text
AEFC-FRL/
|-- aefc/                    # Security, federation, attacks, and CPS surrogate
|-- analysis/                # Result validation, statistics, and figure export
|-- configs/                 # PyTorch, mechanism, and IEEE 39 configurations
|-- experiments/
|   |-- matlab/              # Native model integration and experiment runners
|   |-- run_joint.py         # Reduced-order mechanism experiment
|   `-- run_mechanism_calibration.py
|-- matlab_r2024b/           # Official IEEE 39 example bootstrap
|-- scripts/
|   |-- run_diagnostics.py   # Multi-seed diagnostic runner
|   `-- figures/             # Traffic study and plotting utilities
|-- src/
|   |-- agents/              # Agent, replay memory, and networks
|   |-- algorithms/          # AEFC, FedAvg, and FedProx aggregators
|   |-- attacks/             # Client selection and update attacks
|   |-- envs/                # Gym environment construction
|   |-- utils/               # Logging, metrics, and seeding
|   |-- evaluate.py
|   |-- main.py
|   `-- train.py
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Reproducibility Notes

- Set `--seed` for the main runner and `--seeds` for the diagnostic runner.
- The reduced-order experiment reads all security and attack settings from
  `configs/joint20.json`; its output manifest includes a configuration hash.
- Native IEEE 39 claims must be regenerated from a complete validated matrix by
  `analysis/make_ieee39_results.py`. The official model and raw native outputs
  remain external/generated assets.
- Reported communication percentages are computed relative to dense client
  uploads within this reference implementation.
- Generated logs, diagnostic tables, caches, and traffic-study outputs are
  excluded from version control by `.gitignore`.
- The `src/` learner and `scripts/run_diagnostics.py` are compact implementation
  checks. They are not substitutes for the native IEEE 39-bus evidence path.

## License

This project is released under the MIT License. See `LICENSE` for details.


