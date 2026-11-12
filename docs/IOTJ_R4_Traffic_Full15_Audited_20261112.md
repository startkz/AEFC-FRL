# IoT-J R4 no-oracle traffic: audited deep component reconstruction (2026-11-12)

**Formal GitHub Actions:** [Run 43149666342](https://github.com/startkz/AEFC-FRL/actions/runs/43149666342); frozen training source commit `ef0ed1a3d3f99e3b04a3f746b203f5c5b914ce9f`.

**Acceptance:** 60/60 full-budget cells and 1/1 independent CI audit job passed, independently re-audited from the 60 original per-cell Action ZIP archives (0 errors). Four methods times fifteen independent seeds (11,13,17,19,23,29,31,37,41,43,47,53,59,61,67); 80 rounds × 5 clients × 4 training episodes per client per round = 1,600 train episodes per cell; 25 held-out episodes × 5 clients = 125 held-out per cell. Total 96,000 training and 7,500 held-out episodes. Every episode has a 60-step queue physical trajectory; checkpoints are `.pt`, all data JSONL, actual application uplink bytes recorded per round and SHA256 manifest. Post-attack five-step dwell and right censoring independently recomputed. All 25,875 corresponding seed/client/episode/split exogenous-arrivals schedules match across the four methods.

## Actual fifteen-seed held-out statistics
Numbers are mean [95% seed-bootstrap CI], **not** confidence intervals over held-out episode counts.

| Method | Attack-window total-queue threshold success (%) | Post-attack five-step queue-safe dwell (%) | Mean queue length | Actual application uplink bytes per method-seed |
|---|---|---|---|---|
| AEFC-FRL | 11.31 [8.05,14.93] | 2.61 [0.00,6.51] | 53.16 [50.51,56.05] | 750,400 |
| FedRL | 13.55 [9.33,18.93] | 4.32 [0.00,10.03] | 52.17 [48.47,56.16] | 1,716,800 |
| AEFC-no-adapter | 13.39 [9.81,17.49] | 3.95 [0.96,7.73] | 50.76 [47.51,54.29] | 750,400 |
| AEFC-no-proto | 13.33 [8.91,18.67] | 5.01 [0.00,11.09] | 53.41 [49.46,57.31] | 692,800 |

Paired AEFC-FRL minus FedRL: attack-window success **-2.24 pp** (95% CI [-9.81,+4.48]); postattack 5-step dwell **-1.71 pp** (95% CI [-8.64,+4.53]); mean queue **+0.99** (95% CI [-3.81,+6.30], larger is worse). No stable improvement is established; adverse/mixed component-ablation results are retained without retuning. Sparse vs dense is *not* equal byte.

Training post-attack censored episodes: AEFC-FRL 9,777; FedRL 11,354; AEFC-no-adapter 7,241; AEFC-no-proto 9,927. Held-out censored: 1,826; 1,794; 1,801; 1,781 respectively (out of 1,875 held-out episodes per method).

## Method and scope boundaries

- This is a **newly implemented deep component-level** AEFC-FRL traffic reconstruction using PER, critic, local encoder/actor head, shared backbone, phase-labelled prototypes, advantage selection, final Adam displacement Top-k, and a credibility-based sparse server update. It is **not certified code-identical to the original missing IoT-J AEFC-FRL / Simulink--Python controller**.
- Matched FedRL uses identical deep geometry, replay, local training budget and a **dense unclipped server mean**.
- `AEFC-no-adapter` removes the trainable encoder; it is **not** the IEEE39 `adapter_frozen` variant. `AEFC-no-proto` removes exchange and local prototype loss; it is **not** the IEEE39 `proto_alignment_off` variant.
- The historical no-oracle [Run 41407990103](https://github.com/startkz/AEFC/actions/runs/41407990103) is a five-feature **linear sparse interface** and its original `AEFC-FRL` label is not this deep method. Its 15-seed result is retained separately.
- The original Simulink--Python 120-train heterogeneity matrix and original six-method Simulink figures are **not** reproduced by this traffic experiment. No global cross-domain superiority is claimed.
- The attack-window success event is a threshold crossing during steps 15--31; the *separate* post-attack safe dwell requires five continuous queue-threshold steps after step 31. Missing events before 60-step horizon remain right-censored, not a measured 60-step recovery time.
- Correct journal manuscript LaTeX was not located in the unrelated PATBU/Gate repository branch. Use standalone integrable supplemental LaTeX and separate reviewer response.

### Evidence entry points
- [Raw artifact index / complete CI audit](https://github.com/startkz/AEFC-FRL/actions/runs/43149666342)
- [Source at frozen experiment commit](https://github.com/startkz/AEFC-FRL/tree/ef0ed1a3d3f99e3b04a3f746b203f5c5b914ce9f/experiments/traffic_full)
- [Historical no-oracle diagnostic](https://github.com/startkz/AEFC/actions/runs/41407990103)

The downloadable standalone supplement delivered in the conversation includes full original raw Action ZIPs, per-cell SHA256 inventory, all seed statistics, bilingual response deltas, and compiled LaTeX preview. GitHub source and model artifacts remain separate from manuscript integration.
