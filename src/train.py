from typing import Dict, List
import numpy as np
import torch
from tqdm import trange
from envs.make_env import make_envs
from agents.agent import Agent
from algorithms.aefc import AEFCAggregator
from algorithms.fedavg import FedAvgAggregator
from algorithms.fedprox import FedProxAggregator
from attacks.attacker_selection import select_malicious
from attacks.attack import apply_attack_to_update
from utils.metrics import (
    params_to_vector, vector_to_params, dense_to_sparse, sparse_to_dense,
    compute_comm_percent, moving_average
)
from utils.logger import Logger
from evaluate import evaluate_policy

def build_aggregator(cfg):
    algo = cfg["algorithm"]
    if algo == "AEFC":
        return AEFCAggregator(kappa=cfg["kappa"])
    elif algo == "FedAvg":
        return FedAvgAggregator()
    elif algo == "FedProx":
        return FedProxAggregator()
    else:
        raise ValueError(f"Unknown algorithm {algo}")

def run_training(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = cfg["num_agents"]
    envs = make_envs(cfg["env"], N, max_steps=cfg["max_steps_per_episode"], reward_flip=(cfg["attack_type"]=="reward_flip"))
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    act_high = float(envs[0].action_space.high[0])

    # agents share the same net architecture
    agents: List[Agent] = [
        Agent(
            obs_dim, act_dim, act_high,
            actor_lr=cfg["actor_lr"], critic_lr=cfg["critic_lr"],
            gamma=cfg["gamma"], tau=cfg["tau"],
            batch_size=cfg["batch_size"],
            replay_size=cfg["replay_size"],
            noise_std=cfg["noise_std"],
            device=device,
            prox_mu=(cfg["mu"] if cfg["algorithm"]=="FedProx" else 0.0)
        ) for _ in range(N)
    ]

    # initialize global model from agent[0]
    global_actor = agents[0].actor.clone()
    global_critic = agents[0].critic.clone()
    agg = build_aggregator(cfg)

    rounds = cfg["episodes"] // cfg["sync_interval"]
    malicious_idx = select_malicious(N, frac=cfg["malicious_frac"], seed=cfg["seed"])

    logger = Logger(cfg.get("log_dir","runs"))
    logger.log_config(cfg)
    comm_history = []
    reward_ma = moving_average(window=5)

    step_counter = 0
    for r in trange(rounds, desc="Federated Rounds"):
        # broadcast global params
        for ag in agents:
            ag.sync_with_global(global_actor, global_critic)

        # local training
        client_updates = []
        cred_scores = []
        dense_update_lengths = []

        for i, (ag, env) in enumerate(zip(agents, envs)):
            upd = ag.local_train(env, episodes=cfg["sync_interval"])
            # convert local delta vs current global
            g_vec = params_to_vector(global_actor)
            a_vec = params_to_vector(ag.actor)
            delta = (a_vec - g_vec).detach().cpu()

            if cfg["algorithm"] == "AEFC":
                # advantage list produced during local_train
                credibility = float(max(np.mean(ag.recent_advantages) if ag.recent_advantages else 0.0, 0.0))
                # top-k sparsification
                indices, values = dense_to_sparse(delta, k_ratio=agg.kappa)
                update = {"type":"sparse","indices":indices,"values":values,"num_params":delta.numel(),"cred":credibility}
                cred_scores.append(credibility)
            else:
                update = {"type":"dense","delta":delta,"num_params":delta.numel()}
            # attack simulation
            is_mal = (i in malicious_idx)
            update = apply_attack_to_update(update, attack_type=cfg["attack_type"], is_malicious=is_mal)

            client_updates.append(update)
            dense_update_lengths.append(delta.numel())
            step_counter += upd["steps"]

        # aggregate
        if cfg["algorithm"] == "AEFC":
            new_delta = agg.aggregate(client_updates)
        elif cfg["algorithm"] == "FedAvg":
            new_delta = agg.aggregate(client_updates)
        else:  # FedProx shares FedAvg aggregation; proximal applied in local loss
            new_delta = agg.aggregate(client_updates)

        # update global actor
        with torch.no_grad():
            g_vec = params_to_vector(global_actor).cpu()
            g_vec += new_delta
            vector_to_params(g_vec, global_actor)

        # simple critic sync (DDPG-style global critic = average critics)
        with torch.no_grad():
            # average critic params across clients for stability
            c_accum = None
            for ag in agents:
                c_vec = params_to_vector(ag.critic).cpu()
                c_accum = c_vec if c_accum is None else (c_accum + c_vec)
            c_vec_mean = c_accum / len(agents)
            vector_to_params(c_vec_mean, global_critic)

        # logging: communication & eval
        comm_pct = compute_comm_percent(client_updates, dense_update_lengths)
        comm_history.append(comm_pct)

        if (r+1) % max(1, rounds // 10) == 0 or r == rounds-1:
            eval_res = evaluate_policy(global_actor, envs[0], episodes=cfg["eval_episodes"])
            reward_ma.add(eval_res["avg_reward"])
            logger.log_round(r, eval_res, comm_pct)

    # final evaluation
    final_eval = evaluate_policy(global_actor, envs[0], episodes=cfg["eval_episodes"])
    logger.log_final(final_eval, np.mean(comm_history[-5:]) if len(comm_history) >= 5 else np.mean(comm_history))

    print("Training complete.")
