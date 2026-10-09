import csv
import json
from pathlib import Path

import numpy as np


FEATURE_DIM = 6
ACTION_DIM = 2
PARAM_DIM = FEATURE_DIM * ACTION_DIM


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def features(obs_q, phase, attack_flag, bias):
    scale = 30.0
    return np.array(
        [
            1.0,
            obs_q[0] / scale,
            obs_q[1] / scale,
            float(phase),
            float(attack_flag),
            bias,
        ],
        dtype=float,
    )


def run_episode(w, rng, client_id, horizon=60, train=True):
    q = np.array([8.0 + client_id, 6.0 + 0.5 * client_id], dtype=float)
    phase = 0
    delayed_action = 0
    local_bias = (client_id - 2) / 5.0
    log_grads = []
    rewards = []
    queues = []
    recovery_step = horizon
    recovered = False

    for t in range(horizon):
        attack = 15 <= t < 32
        arrival_base = np.array([2.2 + 0.15 * client_id, 1.8 + 0.1 * (4 - client_id)])
        arrivals = rng.poisson(arrival_base + (np.array([1.4, 0.8]) if attack else 0.0))
        q += arrivals

        obs_q = q.copy()
        if attack:
            # Cross-layer corruption: sensor bias plus delayed actuation.
            obs_q = np.maximum(0.0, obs_q + np.array([4.0, -2.0]) + rng.normal(0, 1.0, 2))

        x = features(obs_q, phase, attack, local_bias)
        probs = softmax(w.reshape(ACTION_DIM, FEATURE_DIM) @ x)
        action = int(rng.choice(ACTION_DIM, p=probs)) if train else int(np.argmax(probs))
        applied_action = delayed_action if attack and (client_id % 2 == 0) else action
        delayed_action = action

        service = np.array([5.4, 1.0]) if applied_action == 0 else np.array([1.0, 5.4])
        q = np.maximum(0.0, q - service)
        phase_change = int(applied_action != phase)
        phase = applied_action

        total_queue = float(np.sum(q))
        reward = -total_queue - 0.6 * phase_change - 2.0 * float(np.max(q) > 35.0)
        rewards.append(reward)
        queues.append(total_queue)

        grad = np.zeros((ACTION_DIM, FEATURE_DIM), dtype=float)
        grad[action] += x
        grad -= probs[:, None] * x[None, :]
        log_grads.append(grad.reshape(-1))

        if attack and not recovered and total_queue < 18.0:
            recovery_step = t - 15 + 1
            recovered = True

    returns = np.zeros(len(rewards), dtype=float)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + 0.97 * g
        returns[i] = g
    advantages = returns - np.mean(returns)
    grad = np.sum(np.asarray(log_grads) * advantages[:, None], axis=0) / len(rewards)
    score = np.sum(np.abs(np.asarray(log_grads)[advantages > np.quantile(advantages, 0.65)]), axis=0)
    return {
        "grad": grad,
        "score": score,
        "return": float(np.sum(rewards)),
        "avg_queue": float(np.mean(queues)),
        "recovery_step": float(recovery_step),
        "accuracy": float(recovered),
    }


def local_update(global_w, rng, client_id, local_episodes=4, lr=0.015):
    grads = []
    scores = []
    returns = []
    for _ in range(local_episodes):
        out = run_episode(global_w, rng, client_id, train=True)
        grads.append(out["grad"])
        scores.append(out["score"])
        returns.append(out["return"])
    grad = np.mean(grads, axis=0)
    score = np.mean(scores, axis=0)
    delta = lr * np.clip(grad, -60.0, 60.0)
    return delta, score, float(np.mean(returns)), float(np.std(returns))


def aggregate_fedrl(updates, malicious=None, clip=None):
    dense = []
    for i, delta in enumerate(updates):
        msg = delta.copy()
        if malicious is not None and i in malicious:
            msg = -3.0 * msg
        if clip is not None:
            msg_norm = np.linalg.norm(msg) + 1e-12
            if msg_norm > clip:
                msg = msg * (clip / msg_norm)
        dense.append(msg)
    return np.mean(dense, axis=0), PARAM_DIM * len(updates)


def aggregate_aefc(updates, scores, returns, stds, kappa=0.35, malicious=None):
    k = max(1, int(PARAM_DIM * kappa))
    sparse = []
    creds = []
    for i, (delta, score, ret, std) in enumerate(zip(updates, scores, returns, stds)):
        idx = np.argsort(score)[-k:]
        msg = np.zeros_like(delta)
        msg[idx] = delta[idx]
        if malicious is not None and i in malicious:
            msg = -3.0 * msg
        msg_norm = np.linalg.norm(msg) + 1e-12
        clip = 0.45
        if msg_norm > clip:
            msg = msg * (clip / msg_norm)
        sparse.append(msg)
        creds.append(max(0.02, np.exp(ret / 2200.0) / (1.0 + std / 100.0)))
    norms = np.asarray([np.linalg.norm(s) for s in sparse], dtype=float)
    norm_ref = np.median(norms) + 1e-12
    norm_gate = np.minimum(1.0, norm_ref / (norms + 1e-12))
    median_update = np.median(np.asarray(sparse), axis=0)
    median_norm = np.linalg.norm(median_update) + 1e-12
    align_gate = []
    for msg in sparse:
        cos = float(np.dot(msg, median_update) / ((np.linalg.norm(msg) + 1e-12) * median_norm))
        align_gate.append(max(0.05, (cos + 1.0) / 2.0))
    creds = np.asarray(creds, dtype=float) * norm_gate * np.asarray(align_gate, dtype=float)
    creds = creds / np.sum(creds)
    return np.sum(np.asarray(sparse) * creds[:, None], axis=0), k * 2 * len(updates)


def evaluate(w, rng, num_clients=5, episodes=25):
    vals = []
    for client_id in range(num_clients):
        for _ in range(episodes):
            vals.append(run_episode(w, rng, client_id, train=False))
    return {
        "avg_queue": float(np.mean([v["avg_queue"] for v in vals])),
        "recovery_step": float(np.mean([v["recovery_step"] for v in vals])),
        "recovery_accuracy": float(np.mean([v["accuracy"] for v in vals]) * 100.0),
    }


def train(method, seed, rounds=80, num_clients=5):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.02, size=PARAM_DIM)
    comm = 0
    malicious = {seed % num_clients}
    for _ in range(rounds):
        updates, scores, returns, stds = [], [], [], []
        for client_id in range(num_clients):
            delta, score, ret, std = local_update(w, rng, client_id)
            updates.append(delta)
            scores.append(score)
            returns.append(ret)
            stds.append(std)
        if method == "FedRL":
            agg, cost = aggregate_fedrl(updates, malicious=malicious)
        elif method == "FedRL-Clip":
            agg, cost = aggregate_fedrl(updates, malicious=malicious, clip=0.45)
        else:
            agg, cost = aggregate_aefc(updates, scores, returns, stds, malicious=malicious)
        w = w + agg
        comm += cost
    eval_rng = np.random.default_rng(seed + 10000)
    out = evaluate(w, eval_rng, num_clients=num_clients)
    out["method"] = method
    out["seed"] = seed
    out["comm_units"] = comm
    return out


def summarize(rows):
    summary = []
    for method in sorted(set(r["method"] for r in rows)):
        group = [r for r in rows if r["method"] == method]
        item = {"method": method, "n": len(group)}
        for metric in ("avg_queue", "recovery_step", "recovery_accuracy", "comm_units"):
            vals = np.asarray([g[metric] for g in group], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(vals))
            item[f"{metric}_std"] = float(np.std(vals, ddof=1))
            item[f"{metric}_median"] = float(np.median(vals))
            item[f"{metric}_q1"] = float(np.percentile(vals, 25))
            item[f"{metric}_q3"] = float(np.percentile(vals, 75))
        summary.append(item)
    return summary


def main():
    out_dir = Path(__file__).resolve().parent
    seeds = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
    rows = []
    for seed in seeds:
        for method in ("FedRL", "FedRL-Clip", "AEFC-FRL"):
            rows.append(train(method, seed))

    csv_path = out_dir / "traffic_poc_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "scenario": "Two-phase traffic-signal queue recovery under sensor bias and command-delay attacks.",
        "scope": "Non-grid proof of concept for the AEFC interface; not a full traffic-control benchmark.",
        "seeds": seeds,
        "summary": summarize(rows),
    }
    json_path = out_dir / "traffic_poc_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
