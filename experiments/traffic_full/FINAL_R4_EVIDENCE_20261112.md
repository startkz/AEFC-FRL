# IoT-J R4: audited no-oracle traffic experiment — 2026-11-12

**Training source**: `ef0ed1a3d3f99e3b04a3f746b203f5c5b914ce9f` on `experiment/traffic-full-no-oracle-20261112`.

**GitHub Actions**: https://github.com/startkz/AEFC-FRL/actions/runs/43149666342 — 61/61 successful jobs (60 seed/method jobs and one audit). An additional independent audit rerun on all 60 original downloaded artifacts returned `passed=true`, `audited_cells=60`, `errors=[]`.

**Budget**: 4 methods x 15 independent seeds = 60 cells; 5 clients; 80 federated rounds; 4 training episodes per client per round; 25 held-out episodes per client; 60 steps per episode. Totals: 96,000 training episodes, 7,500 held-out episodes; 38,299 training and 7,202 held-out right-censored episodes. Each cell has model.pt, JSONL training/evaluation physical traces, 80-round serialized uplink logs, summary.json and manifest SHA256.

## Held-out results (15 training-seed bootstrap 95% confidence intervals)

| Method | Attack-window success | Post-attack 5-step dwell | Mean queue | Application uplink bytes per cell |
|---|---:|---:|---:|---:|
| AEFC-FRL component reconstruction | 11.31% [8.05,14.93] | 2.61% [0,6.51] | 53.16 [50.51,56.05] | 750,400 |
| FedRL dense, unclipped | 13.55% [9.33,18.93] | 4.32% [0,10.03] | 52.17 [48.47,56.16] | 1,716,800 |
| AEFC-no-adapter (remove encoder) | 13.39% [9.81,17.49] | 3.95% [0.96,7.73] | 50.76 [47.51,54.29] | 750,400 |
| AEFC-no-proto (remove alignment and exchange) | 13.33% [8.91,18.67] | 5.01% [0,11.09] | 53.41 [49.46,57.31] | 692,800 |

AEFC-FRL minus FedRL paired seed difference: attack-window success -2.24 percentage points, 95% CI [-9.81,+4.48]; post-attack five-step recovery -1.71 pp, CI [-8.64,+4.53]. **The data do not support a traffic-domain performance superiority claim.** Held-out episodes are not independent training seeds.

The attack-window success checks a queue threshold during steps 15–31 and is **not** post-attack recovery. The second metric requires five consecutive qualifying steps from step 32 onward; missing recovery is right-censored at the 60-step horizon. Uplink bytes count serialized application payload, not lower-level protocol or server downlink.

## Limits and method identity

- The experiment is a **component-level reconstruction** of AEFC-FRL on the historical two-phase Poisson traffic queue simulator: local actor/critic, PER, personalized encoder, shared backbone, prototypes, positive-estimated-advantage update selection, sparse final Adam displacement. Exact source parity with the frozen original IoT-J manuscript is **not verified**.
- It is separate from the historical **linear-sparse** no-oracle diagnostic in `startkz/AEFC` (Run 41407990103; source `1d6bbea00c370a640c12dfcf7df62e5226bbf57e`). Historical success rates: linear sparse 23.57%, FedRL 36.21%, FedRL-Clip 54.19%; do not pool the two experiments.
- `AEFC-no-adapter` **removes** an encoder, unlike prior AC `adapter_frozen`, which freezes training. `AEFC-no-proto` removes **alignment and exchange**, unlike prior `proto_alignment_off`, which only disables local alignment.
- This traffic result is **not** the native Simulink–Python 6-factor-cell x 4-method x 5-seed = 120-training heterogeneity matrix, nor does it reproduce original IEEE39 Simulink figures.
- No global critic-error bound or monotonic improvement theorem is implied. Preserve adverse outcomes and label original model accordingly.

This report is a documentation-only addition. It does **not** modify frozen rewards, seeds, training code, or the other PATBU/Risk Gate manuscript.