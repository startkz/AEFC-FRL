# IEEE 39-bus migration target for GitHub-hosted MATLAB

The original DESL-EPFL model was developed for MATLAB/Simulink R2015aSP1 with ARTEMIS and RT-LAB. This branch evaluates an offline-simulation migration to MATLAB R2022b because GitHub-hosted MATLAB Actions support R2021a or later and R2022b is compatible with Python 3.10 through MATLAB Engine.

The compatibility process is evidence-first. The workflow clones the public DESL-EPFL repository, verifies `git hash-object model.zip` equals `41db586d592851c4a81205a4cc5c7c770b7a0c48`, inventories every Simulink block and library reference, tests `SimulationCommand=update`, saves an R2022b copy, and runs a short physical smoke simulation. No surrogate plant is accepted as IEEE39 evidence.

Only after the probe identifies the actual ARTEMIS/RT-LAB dependencies will the migration replace or remove runtime-only blocks. Generator, transmission-line, transformer, load, breaker, measurement, and powergui blocks are treated as physical-network components and are not removed merely to make the model compile.
