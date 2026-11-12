# IoT-J traffic environment: component-complete AEFC-FRL reconstruction

**Status: implementation and local smoke tests, not yet a 60/60 independently audited experiment.**

This package reconstructs a traffic-domain implementation containing:

- a five-feature **no-oracle** observation, two-phase queue simulator preserving the earlier `Traffic-PoC.py` arrival/service/attack equations;
- per-client actor--critic, prioritized experience replay (PER), personalized encoder and action head, shared actor backbone;
- phase-labelled prototype exchange/alignment, positive estimated-advantage sharing, final Adam displacement Top-k support, and credibility-weighted clipped server aggregation;
- matched-budget FedRL (same actor/critic/backbone, dense aggregation); separate **no-adapter** and **no-prototype** component ablations;
- one sign-flip adversarial client per run, chosen as `seed % 5`, as in the historical linear PoC;
- 80 rounds, five clients, four 60-step training episodes per client/round, 25 held-out episodes per client, fifteen independent seeds;
- exact application-message serialization bytes, 1600 training-episode records and 125 held-out records per seed/method with **all 60 physical steps per episode**, round logs, model.pt and SHA256 manifest.

**Terminology**: The previous `startkz/AEFC` `iotj_bridge/traffic_no_oracle.py` is only a *linear-sparse historical control*, despite its `AEFC-FRL` text label. Neither it nor the new training code should be represented as a verbatim reproduction of an unpublished/unavailable original Simulink--Python implementation. This new reconstruction implements the listed components, but algorithmic parity with the *correct IoT-J AEFC-FRL manuscript and frozen hyperparameters* still requires source-by-source confirmation before journal claims.

## Execute (Python 3.11, CPU)

```bash
python -m pip install numpy==1.26.4
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python experiments/traffic_full/test_traffic.py
python experiments/traffic_full/train.py --method AEFC-FRL --seed 11 --out results/traffic_full/AEFC-FRL/11
```

The GitHub Actions matrix `.github/workflows/traffic-full-no-oracle.yml` is preconfigured for all **4 methods x 15 seeds = 60 independent full-budget cells**. It runs 80 rounds/4 local episodes/25 eval episodes each and requires all 60 cells to pass an independent audit. Workflow jobs and paths are **not** a completed result until Actions plus audit finish.

The independent audit deliberately refuses incomplete or corrupted matrices:

```bash
python experiments/traffic_full/audit.py --root results/traffic_full --report results/traffic_full_audit
python experiments/traffic_full/summarize.py --per-seed results/traffic_full_audit/per_seed.csv --output results/traffic_full_audit
```

## Correct endpoint semantics

The **historical** attack-window success counts a queue below 18 only during steps 15--31. It is *not* post-attack recovery. Missing events are recorded as `-1`. The **new** post-attack safety endpoint is a first five-step dwell with total queue below 18 and no individual queue above 35, from step 32 onward; right-censored at the 60-step horizon. Report both separately. Queue safety here is an illustrative congestion proxy, not a certified CPS invariant.

Each evaluation episode uses a deterministic method-independent stream for exogenous Poisson arrivals and observation noise. No training/held-out episode counts as an independent seed. Confidence intervals are computed across 15 seed-level means with a fixed-seed bootstrap. Methods are not equal-byte: FedRL sends dense updates, AEFC sends sparse index/value/utility and optional prototype metadata; only serialized application bytes are counted (no packet overhead, retransmissions or server downlink).

## Claim discipline

- The new deep traffic results, if eventually audited, must remain separate from the historical linear-sparse interface and the original IEEE39 Simulink-Python figures.
- The no-adapter ablation *removes the trainable encoder* (zero-padding the 5 observed values to 16 latent coordinates); it is **not** the prior `adapter_frozen` ablation.
- The no-prototype ablation removes both local alignment and prototype exchange; it is **not** necessarily the prior `proto_alignment_off` ablation.
- Nonpositive, mixed or adverse results must be retained without retuning rewards or seeds.
- Do not assert monotonic return improvements or uniform critic error bounds from finite-horizon empirical diagnostics.

The independent auditor recomputes all queue state transitions, delayed actions, per-step rewards, congestion and five-step recovery/censoring from the archived physical trace. The FedRL comparator uses **unclipped dense averaging**; sparsity and update clipping are AEFC-specific.