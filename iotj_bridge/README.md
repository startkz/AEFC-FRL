# IoT-J AEFC-FRL cross-repository experiment bridge (work in progress)

This directory is a separate experiment package for IoT-76369-2026.
Its code and data are **not** the bounded-authority AEFC (PATBU/Gate)
results that this repository previously published. Keep those results
and their frozen configurations unchanged.

## Verified reusable environments
- MATLAB R2024b / Simscape Electrical dynamic IEEE39 and validated four
  governor speed/PREF causal interfaces (existing `experiments/matlab`).
- Python 3 AC power-flow standard cases 39 and 118 through pandapower.
  The case118 power flow is **quasi-static** and cannot validate transient
  frequency dynamics. For a dynamic 118-bus claim, another dynamic model
  and observable frequency signals are required.
- Normalized dual-valve hydro SIL (experiment/hydro-valve-cps-20261030);
  it uses PI-parameter updates, not an original AEFC-FRL actor critic.

## Evidence boundary
Neither the physical IEEE39 MATLAB scripts nor the hydro historical
raw traces can be repurposed as learning results for AEFC-FRL. Re-run
the matching local-critic / shared-actor-backbone implementation with
identical reward, attack protocol and optimizer budgets.

The independent `ac_grid_validate.py` program probes power flow and
electrical safety signals for case39 and case118, recording model
provenance in JSON. This is environment validation only.
The full learner, experimental protocol, and preliminary hydro benchmark
are supplied in the companion IoT-J source package until merged here.

## Protocol gates
1. Shared actor backbone updated from detached high positive advantages.
2. Local critics and local observation/action adapters, prototype alignment.
3. Exact application-layer serialized bytes and matched Top-k/ordinary controls.
4. Joint update/utility/prototype metadata forgery and honest distribution shift.
5. Run-level confidence intervals; no claim from smoke/plot arrays.
6. Physical safety: voltages/thermal for AC power flow; rotor-speed/frequency
   only in a verified dynamic simulation.
7. Version every dataset with commit SHA, package versions and full settings.
