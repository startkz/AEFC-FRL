import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from agents.agent import Agent
from agents.networks import Critic
from algorithms.aefc import AEFCAggregator
from attacks.attack import apply_attack_to_update
from attacks.attacker_selection import select_malicious
from envs.make_env import make_envs
from evaluate import evaluate_policy
from utils.metrics import dense_to_sparse, params_to_vector, vector_to_params
from utils.seed import set_global_seed


def hidden_repr(actor, obs_np):
    if obs_np.size == 0:
        return None
    obs = torch.tensor(obs_np, dtype=torch.float32, device=next(actor.parameters()).device)
    with torch.no_grad():
        x = actor.net[0](obs)
        x = actor.net[1](x)
    return x.detach().cpu().numpy()


def state_bins(obs_np):
    angle = np.abs(np.arctan2(obs_np[:, 1], obs_np[:, 0]))
    vel = np.abs(obs_np[:, 2])
    labels = np.full(obs_np.shape[0], 2, dtype=np.int64)
    labels[(angle < 1.2) & (vel < 4.0)] = 1
    labels[(angle < 0.35) & (vel < 1.0)] = 0
    return labels


def local_prototypes(agent):
    n = len(agent.replay)
    if n == 0:
        return {}
    obs = agent.replay.obs[:n].copy()
    labels = state_bins(obs)
    reps = hidden_repr(agent.actor, obs)
    if reps is None:
        return {}
    out = {}
    for k in (0, 1, 2):
        mask = labels == k
        if np.any(mask):
            out[k] = reps[mask].mean(axis=0)
    return out


def cosine_drift(local_proto, global_proto):
    vals = []
    for k, p in local_proto.items():
        if k not in global_proto:
            continue
        q = global_proto[k]
        denom = (np.linalg.norm(p) * np.linalg.norm(q)) + 1e-12
        vals.append(1.0 - float(np.dot(p, q) / denom))
    return float(np.mean(vals)) if vals else 0.0


def mean_global_proto(local_list):
    grouped = {}
    for protos in local_list:
        for k, p in protos.items():
            grouped.setdefault(k, []).append(p)
    return {k: np.mean(v, axis=0) for k, v in grouped.items()}


def sparse_to_dense_update(update):
    out = torch.zeros(update["num_params"], dtype=torch.float32)
    if update["type"] == "dense":
        return update["delta"].detach().cpu()
    out[update["indices"]] = update["values"].detach().cpu()
    return out


def clone_critic(critic):
    obs_act_dim = critic.net[0].in_features
    obs_dim = obs_act_dim - 1
    new = Critic(obs_dim, 1)
    new.load_state_dict(critic.state_dict())
    return new


def equal_sparse_average(client_updates):
    dense = [sparse_to_dense_update(u) for u in client_updates]
    return torch.stack(dense, dim=0).mean(dim=0)


def run_one(seed, args):
    set_global_seed(seed)
    device = torch.device("cpu")
    envs = make_envs(args.env, args.num_agents, max_steps=args.max_steps, reward_flip=False)
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    act_high = float(envs[0].action_space.high[0])

    agents = [
        Agent(
            obs_dim,
            act_dim,
            act_high,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            tau=args.tau,
            batch_size=args.batch_size,
            replay_size=args.replay_size,
            noise_std=args.noise_std,
            device=device,
        )
        for _ in range(args.num_agents)
    ]
    global_actor = agents[0].actor.clone()
    global_critic = clone_critic(agents[0].critic)
    aggregator = AEFCAggregator(kappa=args.kappa)
    malicious_idx = select_malicious(args.num_agents, frac=args.malicious_frac, seed=seed)
    rounds = args.episodes // args.sync_interval
    prev_global_proto = None
    rows = []

    for round_idx in range(rounds):
        for ag in agents:
            ag.sync_with_global(global_actor, global_critic)

        client_updates = []
        dense_deltas = []

        for i, (ag, env) in enumerate(zip(agents, envs)):
            ag.local_train(env, episodes=args.sync_interval)
            g_vec = params_to_vector(global_actor)
            a_vec = params_to_vector(ag.actor)
            delta = (a_vec - g_vec).detach().cpu()
            dense_deltas.append(delta)

            credibility = float(max(np.mean(ag.recent_advantages) if ag.recent_advantages else 0.0, 0.0))
            indices, values = dense_to_sparse(delta, k_ratio=args.kappa)
            update = {
                "type": "sparse",
                "indices": indices,
                "values": values,
                "num_params": delta.numel(),
                "cred": credibility,
            }
            update = apply_attack_to_update(
                update,
                attack_type=args.attack_type,
                is_malicious=(i in malicious_idx),
            )
            client_updates.append(update)

        new_delta = aggregator.aggregate(client_updates)
        equal_delta = equal_sparse_average(client_updates)
        dense_stack = torch.stack(dense_deltas, dim=0)
        mean_dense = dense_stack.mean(dim=0)

        g_t = float(torch.linalg.vector_norm(new_delta).item())
        h_abs = float(torch.linalg.vector_norm(dense_stack - mean_dense, dim=1).mean().item())
        h_rel = float(h_abs / (torch.linalg.vector_norm(mean_dense).item() + 1e-12))
        b_abs = float(torch.linalg.vector_norm(new_delta - equal_delta).item())
        b_rel = float(b_abs / (g_t + 1e-12))

        local_proto = [local_prototypes(ag) for ag in agents]
        if prev_global_proto is None:
            s_t = 0.0
        else:
            s_t = float(np.mean([cosine_drift(p, prev_global_proto) for p in local_proto]))
        prev_global_proto = mean_global_proto(local_proto)

        with torch.no_grad():
            g_vec = params_to_vector(global_actor).cpu()
            g_vec += new_delta
            vector_to_params(g_vec, global_actor)
            c_vecs = [params_to_vector(ag.critic).cpu() for ag in agents]
            vector_to_params(torch.stack(c_vecs, dim=0).mean(dim=0), global_critic)

        if (round_idx + 1) % max(1, rounds // 4) == 0 or round_idx == rounds - 1:
            eval_res = evaluate_policy(global_actor, envs[0], episodes=args.eval_episodes)
            avg_reward = float(eval_res["avg_reward"])
        else:
            avg_reward = math.nan

        rows.append(
            {
                "seed": seed,
                "round": round_idx + 1,
                "G_t": g_t,
                "H_t_abs": h_abs,
                "H_t_rel": h_rel,
                "B_t_abs": b_abs,
                "B_t_rel": b_rel,
                "S_t": s_t,
                "avg_reward": avg_reward,
            }
        )
    for env in envs:
        env.close()
    return rows


def summarize(rows):
    metrics = ["G_t", "H_t_rel", "B_t_rel", "S_t"]
    stable_rows = [r for r in rows if int(r["round"]) > 1]
    out = {}
    for m in metrics:
        vals = np.array([float(r[m]) for r in stable_rows], dtype=float)
        out[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "max": float(vals.max()),
            "p95": float(np.percentile(vals, 95)),
            "n": int(len(vals)),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--num_agents", type=int, default=5)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--sync_interval", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--eval_episodes", type=int, default=3)
    p.add_argument("--kappa", type=float, default=0.2)
    p.add_argument("--malicious_frac", type=float, default=0.1)
    p.add_argument("--attack_type", default="random", choices=["none", "random", "poison"])
    p.add_argument("--actor_lr", type=float, default=1e-3)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--replay_size", type=int, default=2000)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--out_dir", default="diagnostics")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    all_rows = []
    for seed in seeds:
        all_rows.extend(run_one(seed, args))

    csv_path = out_dir / "aefc_diagnostics_rounds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "config": vars(args),
        "summary": summarize(all_rows),
        "notes": {
            "environment": args.env,
            "scope": "Released reference implementation diagnostic run; not a Simulink power-grid run.",
            "S_t": "Post-hoc hidden-representation prototype drift computed from replay states.",
        },
    }
    json_path = out_dir / "aefc_diagnostics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
