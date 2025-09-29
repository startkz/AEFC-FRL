# API Notes

## Aggregators

### AEFCAggregator(kappa: float)
- `aggregate(client_updates) -> torch.Tensor`
  - Each client update is a dict:
    - Sparse: `{"type":"sparse","indices":LongTensor,"values":FloatTensor,"num_params":int,"cred":float}`
    - Dense : `{"type":"dense","delta":FloatTensor,"num_params":int}`
  - Computes weights from `cred` (non-negative), reconstructs dense global delta via weighted sum.
  - Falls back to simple averaging if all cred ≤ 0.

### FedAvgAggregator()
- `aggregate(client_updates)` – averages dense deltas; sparse are densified then averaged.

### FedProxAggregator()
- Same aggregation as FedAvg; proximal term is handled inside `Agent` local loss.

## Agent

### Agent.local_train(env, episodes:int)
- Runs local DDPG-style training for `episodes`, stores recent TD errors as advantage proxy.
- Returns `{"steps": int}`.

### Agent.sync_with_global(actor, critic)
- Copies global nets into local nets, keeps target nets in sync.

## Attacks

### attacker_selection.select_malicious(num_clients, frac, seed)
- Returns a set of malicious agent indices.

### attack.apply_attack_to_update(update, attack_type, is_malicious)
- Modifies update (random or poison) if the client is malicious.

## Metrics

- `params_to_vector`, `vector_to_params` to flatten/unflatten model parameters.
- `dense_to_sparse` top-k magnitude selection with ratio `kappa`.
- `compute_comm_percent` returns % of baseline (dense) comm cost.

