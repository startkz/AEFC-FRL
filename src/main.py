import argparse
import yaml
from pathlib import Path
from train import run_training
from utils.seed import set_global_seed

def parse_args():
    p = argparse.ArgumentParser(description="AEFC-FRL Federated RL")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    # core
    p.add_argument("--algorithm", type=str, default="AEFC", choices=["AEFC","FedAvg","FedProx"])
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--num_agents", type=int, default=10)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--sync_interval", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=20)
    # AEFC / FedProx
    p.add_argument("--kappa", type=float, default=0.2)
    p.add_argument("--mu", type=float, default=0.0)
    # attacks
    p.add_argument("--malicious_frac", type=float, default=0.0)
    p.add_argument("--attack_type", type=str, default="none", choices=["none","random","poison","reward_flip"])
    # RL
    p.add_argument("--actor_lr", type=float, default=1e-3)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--replay_size", type=int, default=50000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--max_steps_per_episode", type=int, default=200)
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="runs")
    return p.parse_args()

def merge_config(args):
    if args.config is None:
        return vars(args)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    # CLI overrides YAML
    for k, v in vars(args).items():
        if k == "config": 
            continue
        if (k not in cfg) or (v != parse_args().__dict__[k]):
            cfg[k] = v
    return cfg

if __name__ == "__main__":
    args = parse_args()
    cfg = merge_config(args)
    set_global_seed(cfg.get("seed", 42))
    Path(cfg.get("log_dir","runs")).mkdir(parents=True, exist_ok=True)
    run_training(cfg)
