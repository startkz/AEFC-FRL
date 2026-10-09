#!/usr/bin/env python3
import csv, hashlib, json, os
from pathlib import Path
import numpy as np

from aefc.security import (Evidence, PATBUConfig, ProvenanceAwareTemporalBelief,
                           GateConfig, RiskAwareRecoveryGate, AdaptConfig,
                           TrustGatedAdaptation, ShieldConfig, RobustBoxShield)
from aefc.attacks import AttackConfig, observation_attack, post_shield_attack, poison_update, active
from aefc.federation import robust_sparse_aggregate
from aefc.surrogate import ReducedOrderCPS

METHODS = ["FedRL", "FedRL-Clip", "RobustFedRL-Clip", "AEFC-Full"]
SCENARIO = "joint"


def h(v):
    return hashlib.sha256(np.asarray(v, dtype=float).tobytes()).hexdigest()[:16]


def clipped_update(theta, agg, eta, clip_norm):
    g = np.asarray(agg, dtype=float).copy()
    n = np.linalg.norm(g)
    if n > clip_norm:
        g *= clip_norm / n
    return theta - eta * g


def run(seed, method, out_dir, cfg, steps, nclients, f, config_id):
    # Crucial fairness property: RNG seed excludes method name. Each method with
    # the same seed consumes the same random stream for observations, client
    # updates, poisoning, and downstream physical corruption.
    rng = np.random.default_rng(seed * 1009 + 424242)
    env = ReducedOrderCPS(dim=4, seed=seed)
    x = env.reset()
    patbu_cfg = dict(cfg.get("patbu", {})); patbu_cfg["trim_f"] = f
    patbu = ProvenanceAwareTemporalBelief(4, PATBUConfig(**patbu_cfg))
    gate = RiskAwareRecoveryGate(GateConfig(**cfg.get("gate", {})))
    adapt = TrustGatedAdaptation(4, AdaptConfig(**cfg.get("adaptation", {})))
    shield_cfg = dict(cfg.get("shield", {})); shield_cfg["state_limit"] = float(cfg.get("safe_state_limit", 0.35))
    shield = RobustBoxShield(ShieldConfig(**shield_cfg))
    atk = AttackConfig(**cfg.get("attack", {}))
    common_k = int(cfg.get("common_support_k", 3))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"seed_{seed:02d}.jsonl"

    violation = False; recovery_time = None; interventions = 0; false_auth = 0; dangerous = 0
    max_belief_dist = 0.0; max_policy_drift = 0.0; min_margin = 1e9; agg_dev = []
    clean_belief = np.zeros(4); clean_theta = np.zeros(4)

    with trace_path.open("w", encoding="utf-8") as fp:
        for t in range(steps):
            clean_obs = env.clean_observation()
            obs = clean_obs + observation_attack(4, t, rng, atk, True)
            client_updates = []; evidence = []; honest = []
            for j in range(nclients):
                g = 0.65 * x + 0.15 * rng.normal(size=4)
                honest.append(g.copy())
                byz = active(t, atk) and j < f
                gu = poison_update(g, t, rng, atk, byz)
                client_updates.append(gu)
                ts = t - atk.stale_delay if byz else t
                evv = -0.55 * x + 0.08 * rng.normal(size=4)
                if byz:
                    evv = 2.8 * evv + 0.6
                evidence.append(Evidence(evv, f"client-{j}", ts, 1, f"round-{t}", provenance_valid=not byz))

            if method in ("FedRL", "FedRL-Clip"):
                agg = np.mean(np.vstack(client_updates), axis=0); support = np.arange(4); trust = 0.5
            else:
                agg, support = robust_sparse_aggregate(client_updates, f=f, k=common_k); trust = 0.5

            clean_k = np.mean(np.vstack([e.value for e in evidence if e.provenance_valid]), axis=0)
            clean_belief = 0.72 * clean_belief + 0.20 * clean_obs + 0.08 * clean_k
            if method == "AEFC-Full":
                belief, diag = patbu.update(obs, evidence, t); trust = diag["trust"]
            else:
                belief = obs.copy(); trust = 0.5
            belief_dist = float(np.linalg.norm(belief - clean_belief))
            max_belief_dist = max(max_belief_dist, belief_dist)

            proposal = -0.55 * belief + 1.5 * adapt.deployed
            safety_shortfall = max(0.0, np.max(np.abs(x + env.drift() + 0.24 * proposal)) - 0.30)
            recovery_shortfall = max(0.0, np.linalg.norm(x, ord=np.inf) - 0.08)
            uncertainty = min(1.0, belief_dist)
            gate_allowed, risk = gate.assess(safety_shortfall, recovery_shortfall, uncertainty, trust)
            true_danger = np.max(np.abs(x + env.drift() + 0.24 * proposal)) > 0.35
            dangerous += int(true_danger)
            if method != "AEFC-Full":
                gate_allowed = True
            false_auth += int(gate_allowed and true_danger)
            if not gate_allowed:
                proposal = -0.35 * x

            if method == "FedRL":
                adapt.deployed = adapt.deployed - 0.12 * agg
                adapt_accepted = True
            elif method in ("FedRL-Clip", "RobustFedRL-Clip"):
                adapt.deployed = clipped_update(adapt.deployed, agg,
                                                float(cfg["adaptation"]["eta"]),
                                                float(cfg["adaptation"]["clip_norm"]))
                adapt_accepted = True
            else:
                cand = adapt.propose(agg, trust)
                val_before = float(np.linalg.norm(x) + 0.08 * np.linalg.norm(adapt.deployed))
                val_after = float(np.linalg.norm(x) + 0.08 * np.linalg.norm(cand))
                _, ad_diag = adapt.deploy(cand, trust, gate_allowed, val_before, val_after)
                adapt_accepted = bool(ad_diag.get("accepted", False))

            policy_drift = float(np.linalg.norm(adapt.deployed - clean_theta))
            max_policy_drift = max(max_policy_drift, policy_drift)

            if method == "AEFC-Full":
                ustar, sh = shield.project(x, proposal, env.drift())
            else:
                ustar = np.clip(proposal, -0.45, 0.45)
                sh = {"feasible": True, "intervention_norm": 0.0,
                      "robust_margin": float(0.35 - np.max(np.abs(x + env.drift() + 0.24 * ustar)))}
            interventions += int(sh["intervention_norm"] > 1e-8)
            min_margin = min(min_margin, sh["robust_margin"])
            au = post_shield_attack(4, t, rng, atk, True)
            uexec = ustar + au
            x = env.step(uexec)
            safe = env.safe(); violation = violation or (not safe)
            if recovery_time is None and t > atk.end and env.recovered():
                recovery_time = t - atk.end

            honest_center = np.mean(np.vstack(honest), axis=0)
            ad = float(np.linalg.norm(agg - honest_center)); agg_dev.append(ad)
            row = {
                "commit_sha": os.environ.get("GITHUB_SHA", "LOCAL"), "config_id": config_id,
                "common_random_numbers": True, "seed": seed, "method": method, "scenario": SCENARIO, "t": t,
                "state_true": x.tolist(), "observation_received": obs.tolist(), "belief_state": belief.tolist(),
                "belief_clean_reference": clean_belief.tolist(), "trust_score": trust, "risk_upper_bound": risk,
                "gate_decision": bool(gate_allowed), "deployed_policy_hash": h(adapt.deployed),
                "robust_aggregate": np.asarray(agg).tolist(), "aggregate_deviation": ad,
                "proposal_action": np.asarray(proposal).tolist(), "shield_action": np.asarray(ustar).tolist(),
                "post_shield_attack": np.asarray(au).tolist(), "executed_action": np.asarray(uexec).tolist(),
                "safe": bool(safe), "belief_distortion": belief_dist, "policy_drift": policy_drift,
                "adapt_accepted": adapt_accepted,
            }
            fp.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "commit_sha": os.environ.get("GITHUB_SHA", "LOCAL"), "config_id": config_id,
        "common_random_numbers": True, "seed": seed, "method": method, "scenario": SCENARIO,
        "safety_violation": int(violation), "min_cbf_margin": min_margin,
        "false_authorization_rate": false_auth / max(dangerous, 1),
        "max_belief_distortion": max_belief_dist, "max_policy_drift": max_policy_drift,
        "recovery_success": int(recovery_time is not None),
        "recovery_time": recovery_time if recovery_time is not None else steps - atk.end,
        "shield_intervention_ratio": interventions / steps,
        "mean_aggregate_deviation": float(np.mean(agg_dev)),
        "backend": "reduced_order_mechanism_fairness_calibration",
    }
    (out_dir / f"seed_{seed:02d}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    cfg_path = Path("configs/joint20.json")
    cfg = json.loads(cfg_path.read_text())
    config_id = hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16] + "-crn"
    seeds = 20; steps = int(cfg.get("steps", 120)); nclients = int(cfg.get("clients", 7)); f = int(cfg.get("byzantine_clients", 2))
    root = Path("results/supplemental_mechanism/calibration")
    rows = []
    for method in METHODS:
        od = root / method
        for seed in range(seeds):
            rows.append(run(seed, method, od, cfg, steps, nclients, f, config_id))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest = {
        "backend": "reduced_order_mechanism_fairness_calibration",
        "commit_sha": os.environ.get("GITHUB_SHA", "LOCAL"), "config_id": config_id,
        "common_random_numbers": True, "seeds": seeds, "steps": steps,
        "methods": METHODS, "scenario": SCENARIO, "rows": len(rows),
        "purpose": "Calibrate the extreme FedRL mechanism-level contrast using matched disturbance streams and tuned clipping baselines.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
