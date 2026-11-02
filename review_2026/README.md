# IoT-J reproducibility gates (working branch)

This branch records the protocol checks required before claiming that an experiment
reproduces the actor-backbone AEFC manuscript. It does not report new experimental
findings and does not modify the original protected main branch.

## Method identity

The manuscript shares recovery-advantage-conditioned actor-backbone deltas while
keeping critics and subsystem adapters local. The existing PyTorch
`src/algorithms/aefc.py` accepts sparse weights and reported credibility, but
does not implement the manuscript's prototype drift and separate advantage
sharing loss. The `aefc/security.py` family is a distinct PATBU/Gate mechanism.
Neither may be labeled a full reproduction without implementation alignment.

## Evidence required

1. Real executed native model, verified grid mapping and scenario/seed manifest.
2. Matched ordinary-loss Top-kappa vs. advantage sharing with optimizer, loss
   budget, serialized uplink and clients held fixed.
3. Adaptive joint update/utility/prototype poisoning and honest operating shifts.
4. Voltage/line/actuation safety traces; frequency only when supported by dynamics.
5. Independent 39/118-bus direction diagnostics with fixed-surrogate reference.
6. Actual serialized application bytes, full run seed matrix and censoring outcomes.

The analysis tool checks structural validity of per-run logs. It is not proof of
full AEFC algorithm identity, safety, statistical significance or robust learning.
Published figure-generation arrays and synthetic smoke fixtures are not empirical
evidence for new manuscript claims.
