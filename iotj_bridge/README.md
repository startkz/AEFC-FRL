# IoT-J 76369 cross-repository learner branch

Canonical physical testbeds and executed GitHub Actions are maintained on
[startkz/AEFC / experiment/iotj-76369-ac-grid-bridge-20261102](https://github.com/startkz/AEFC/tree/experiment/iotj-76369-ac-grid-bridge-20261102).
This directory mirrors the matching developer-stage local critic and shared
actor-backbone learning source to keep **AEFC-FRL** the algorithm-development
repository. Files were copied from the above branch, not merged into main.

## Scope and limitations
- `core.py`: local critic, observation/action networks, advantage-driven
  backbone sharing, Top-k sparse messages, prototype alignment, utility/drift
  weights, and an explicitly optional capped-weight mitigation.
- `run_experiment.py`: Python/pandapower quasi-static 39/118 AC testbed;
  run from AEFC repository for provenance.
- `frozen_diagnostics.py`: local fixed-surrogate Adam/compression/weight
  direction bias/variance empirical snapshot.
- `ac_grid_validate.py`: case provenance and voltage/thermal observables.

The present bridge is **not** the complete submitted PER/target-network
AEFC-FRL and is not interchangeable with the frozen MATLAB IEEE39,
hydropower PATBU/Gate, or existing PyTorch reference learners. Its AC power
flow does **not** model generator frequency transients or formal physical
safety guarantees. As tested, ordinary Top-k and positive-advantage
sharing show little separation while self-reported credibility metadata
is vulnerable to coordinated forgery.

Publication claims require a full matched implementation, independently
trained seeds, defined safe-recovery stopping, adaptive attack optimization,
clean/baseline reconstruction, data provenance, and complete run-level
statistics. See `review_2026/README.md` and the IoT-J response ledger.
