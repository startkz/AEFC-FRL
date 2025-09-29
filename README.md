# AEFC-FRL (PyTorch 2.0, Python 3.9)

Reproducible reference implementation of **Advantage-Execution + Federated-Critic Federated RL (AEFC-FRL)** and baselines **FedAvg** / **FedProx**. Includes:
- Full training/testing pipeline for federated RL
- Attack simulation (random update / gradient poisoning) at 10% or 30% malicious rates
- Example CPS-like environment via Gym (`Pendulum-v1`)
- Communication overhead & accuracy metrics consistent with the paper

> **Python 3.9**, **PyTorch 2.0** recommended. GPU optional.

## 1) Setup

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## 2) Quickstart (AEFC-FRL, 10 agents, 10% random attackers)
python src/main.py \
  --algorithm AEFC --env Pendulum-v1 --num_agents 10 --episodes 500 \
  --sync_interval 5 --malicious_frac 0.1 --attack_type random --kappa 0.2 \
  --seed 42

3) Using YAML config

python src/main.py --config configs/example_config.yaml


AEFC_FRL_Project/
  README.md
  requirements.txt
  configs/
    example_config.yaml
  src/
    main.py
    train.py
    evaluate.py
    algorithms/
      aefc.py
      fedavg.py
      fedprox.py
    agents/
      agent.py
      memory.py
      networks.py
    envs/
      cps_env.py
      make_env.py
    attacks/
      attack.py
      attacker_selection.py
    utils/
      logger.py
      metrics.py
      seed.py
  docs/
    API.md


