# 3d-LQG
Attempting to construct a three dimensional analog for Liouville Quantum Gravity

The project studies Budd–Castro mated-CRT **graphs** and their intrinsic geometry,
alongside a separate Euclidean 3D log-field/GMC/LFPP pipeline. Start with the
latest [Phase 3A comparison](results/RESEARCH_REPORT_PHASE_3A.md),
the [Phase 2C sampler report](results/RESEARCH_REPORT_PHASE_2C.md)
or the [original metric study](results/RESEARCH_REPORT_PHASES_1_2.md).

The independent-coordinate sampler produces the paper's biased excursion law
and its exact continuous-path cell minima. A special correlated Weyl-cone
sampler has exact grid values and approximate minima. The general SPD sampler
is a finite-grid MCMC approximation; its broad dimension-study validation gate
remains closed. These distinctions are recorded in every sample's metadata.

## Setup and tests

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
```

The reported environment used Python 3.14.6 on macOS arm64. The direct package
versions are in `requirements-lock.txt`; `requirements.txt` gives broader
compatible requirements. Plotting uses a noninteractive Matplotlib backend.

## Generate a graph

Run Python from the repository root:

```python
import numpy as np
from src.brownian_excursions import sample_excursion
from src.mated_crt import graph_from_excursion, validate_graph

rng = np.random.default_rng(1234)
sample = sample_excursion(4096, np.eye(3), rng, law="biased")
graph = graph_from_excursion(sample, 4096)
print(validate_graph(graph))
simple_csr = graph.to_sparse()
multigraph_conductances = graph.to_sparse(multigraph=True)
```

Use `src.covariance.from_correlations(a,b,c)`, with `a=C12,b=C13,c=C23`, or
`from_angles(alpha,beta,theta_angle)` in radians. For a general covariance,
`sample_excursion(..., law="unbiased", oversample=8, sweeps=128)` selects the
explicitly approximate finite-grid law. It does not silently replace the
paper's general biased sampler. The exact special correlated sampler is
`src.weyl_excursions.sample_weyl_excursion`.

## Reproduce experiments

The raw records already exist under `results/`. To generate new independent
copies without overwriting them, choose new output filenames:

```bash
.venv/bin/python -m experiments.reproduce_budd_castro --sizes 2048 4096 8192 16384 32768 65536 131072 --graphs 64 --roots 8 --dimension 3 --law biased --seed 2026111001 --output results/reproduction/i3_biased.jsonl
.venv/bin/python -m experiments.reproduce_budd_castro --sizes 32768 65536 131072 262144 524288 --graphs 24 --roots 8 --dimension 3 --law biased --seed 2026111004 --output results/reproduction/i3_biased_high.jsonl
.venv/bin/python -m experiments.reproduce_budd_castro --sizes 2048 4096 8192 16384 32768 65536 --graphs 48 --roots 8 --dimension 2 --law biased --seed 2026111002 --output results/reproduction/i2_biased.jsonl
.venv/bin/python -m experiments.reproduce_budd_castro --sizes 2048 4096 8192 16384 32768 65536 --graphs 48 --roots 8 --dimension 3 --law unbiased --seed 2026111003 --output results/reproduction/i3_unbiased.jsonl
.venv/bin/python -m experiments.combine_studies results/reproduction/i3_biased.jsonl results/reproduction/i3_biased_high.jsonl --output results/reproduction/i3_biased_combined.jsonl --expect 448 120 --seed 941700
```

Each realization has separate deterministic graph and metric seeds. The
JSONL records preserve per-root distance histograms, graph statistics, path
diagnostics, and seeds. Metadata sidecars preserve software/source versions.
Selected full graph edge lists are saved as compressed NPZ files; the other
graphs can be regenerated from their seeds. Sampling is sequential and memory
scales with the largest individual graph.

To regenerate plots and bootstrap analyses from existing raw data:

```bash
.venv/bin/python -m experiments.reproduce_budd_castro --analyze-only --seed 2026111001 --output results/data/i3_biased.jsonl
.venv/bin/python -m experiments.analyze_corrections results/data/i3_biased.jsonl results/data/i2_biased.jsonl results/data/i3_unbiased.jsonl
.venv/bin/python -m experiments.build_tables
```

Other executed scripts are `experiments.validate_sampler`, `cone_diagnostics`,
`parameter_sweep`, `sampler_sensitivity`, `weyl_resolution`, `weyl_study`,
`additional_geometry`, and `benchmark`. Their default seeds/configurations are
in the scripts and saved metadata. Several preserve raw data by refusing to
overwrite an existing output: select a new output directory where the CLI
supports it, or run from a separate checkout. The covariance pilot deliberately
does not fit dimensions. Full methodological details:

- [Mathematical definitions](docs/MATHEMATICAL_METHODS.md)
- [Sampler and graph construction](docs/CONSTRUCTION.md)
- [Special correlated sampler](docs/WEYL_CONSTRUCTION.md)
- [Independent validation](docs/VALIDATION.md)
- [Metric analysis](docs/METRIC_METHODOLOGY.md)
- [General-covariance limitations](docs/GENERAL_COVARIANCE_DIAGNOSTICS.md)

Agent instructions and persistent context remain in `Agents_Info/`; the root
`AGENTS.md` makes them discoverable. Original source PDFs are listed in
[papers/README.md](papers/README.md).

## Phase 2B audit and correlated sampling

The [Phase 2B report](results/RESEARCH_REPORT_PHASE_2B.md) audits the earlier
results, adds 80 independently seeded large I3 graphs, and implements the
paper's biased interior-ball construction for analytic wedges and birectangular
cones. The updated I3 offset fit is 4.910 with MC 95% interval [4.643,5.159];
correction-model and fitting-window uncertainty remains substantial.

General correlated sampling remains unvalidated. The report documents an
interior-zero defect in the original angular FEM interpolation, a connected
mesh repair, and remaining boundary/harmonic errors. Analytic Doob paths also
require tighter joint-minimum and finite-step controls. A large general
covariance sweep was therefore not performed.

New code and methods:

- `src/doob_excursions.py`: explicitly approximate biased Doob/interior-ball paths.
- `src/cone_sampling.py`: cached interpolation and optional connected corner repair.
- [Independent audit](docs/AUDIT_PHASE_2B.md) and [Doob methods](docs/DOOB_SAMPLER.md).
- [Correlated validation](docs/CORRELATED_VALIDATION_PHASE2B.md), including
  matching-law controls, permutations, resolution, failures and error budgets.

New records and metadata are under `results/phase2b/`. Existing results are
preserved. The `phase2b_finalize_analysis`, `phase2b_weight_audit`, and
`phase2b_plot_fem_audit` experiment modules reuse saved numerical results;
their default outputs refuse overwriting. `validate_doob --help` documents
the separately saved control and refinement runs. This work stops before
field/GMC/LFPP comparisons.

## Phase 2C: A3, corner harmonics and path accuracy

The [Phase 2C report](results/RESEARCH_REPORT_PHASE_2C.md) adds an A3 Weyl
benchmark from four noncolliding particles, independent corner-aware angular
methods, path-likelihood diagnostics and improved Brownian ball clocks.
The complete arbitrary-covariance sampler remains unvalidated: full-path and
joint-minimum errors do not meet the recorded accuracy goals. No generic
graph sweep, new dimension fit or field/GMC/LFPP implementation was begun.

New components and their qualifications:

- [A3 derivation](docs/WEYL_A3_PHASE2C.md): `src.weyl_a3.sample_a3_excursion`
  supports separate biased/unbiased laws. Grid values have the exact joint
  law; continuous joint minima remain approximated.
- [Spectral corner method](docs/CORNER_METHODS_PHASE2C.md) and
  [positive-factor Ritz method](docs/CORNER_RITZ_PHASE2C.md): independent
  eigenfunction comparisons, boundary asymptotics and positivity checks.
- [Likelihood audit](docs/LIKELIHOOD_PHASE2C.md) and
  [path validation](docs/PATH_ACCURACY_PHASE2C.md): separate harmonic,
  clock, interpolation and graph errors.

All saved experiments are under `results/phase2c/`; the
[completion checkpoint](results/phase2c/STATUS.md) and
[final manifest](results/phase2c/final_manifest.json) index the handoff.
The scripts `validate_a3`, `validate_corner_solver`, `compare_corner_methods`,
`phase2c_ritz`, `phase2c_likelihood`, and `phase2c_path_validation` under
`experiments/` reproduce the components. Consult their `--help` and saved
metadata, and use a new output directory rather than replacing saved results.
Run the tests with `.venv/bin/python -m pytest -q`.

## Phase 3A: Euclidean fields, GMC, LFPP and intrinsic comparisons

The [Phase 3A report](results/RESEARCH_REPORT_PHASE_3A.md) gives **Outcome B:
partial or ambiguous correspondence**. It includes real coupled ensembles,
independent confirmation and finite-size checks. The I3 candidate
xi=.3,gamma=1.5 is interesting, but parameter calibration and higher-point
statistics remain unresolved. This does not identify a common continuum model.

- [Parameter/source protocol](docs/COMPARISON_PROTOCOL_PHASE3A.md).
- [Field and GMC](docs/FIELD_GMC_PHASE3A.md): unit-log torus spectrum,
  coupled resolutions, exact finite-grid mass variance and heavy-tail audit.
- [LFPP](docs/LFPP_PHASE3A.md): cube/torus, exact lattice vertex crossings,
  6/26/98-direction metrics, isotropy and quadrature controls.
- [Comparison statistics](docs/STATISTICS_PHASE3A.md): equal-realization
  normalizations, roots/balls/matrices, seed split and cluster bootstrap.
- [Results and preservation manifest](results/phase3a/final_manifest.json).

Core reproduction commands below use **new** output directories:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3a_ensembles field --output results/reproduction_phase3a/discovery --sizes 16 24 32 --samples 24 --stencils axis 26 --crossings
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3a_ensembles graph --output results/reproduction_phase3a/graphs --sizes 2048 8192 --samples 48 --q 32 --refine-samples 4
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3a_compare select --graphs results/reproduction_phase3a/graphs --discovery results/reproduction_phase3a/discovery --output results/reproduction_phase3a/selection
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.phase3a_ensembles field --output results/reproduction_phase3a/confirmation --sizes 24 32 48 --samples 24 --seed 202611113201 --pairs 0:0,0.2:0,0.2:0.5,0.2:1,0.3:1.5 --stencils 26 primitive2 --crossings
```

Saved data, exact configurations, full fields, sampled distance matrices,
figures, analysis scripts and source snapshots are under `results/phase3a/`.
The 98-direction confirmation cost about 16 minutes on the recorded machine;
benchmark before expanding it. Analysis modules reuse existing checkpoints
and refuse overwriting completed outputs. Some auxiliary controls have fixed
output paths; reproduce those in a separate checkout or change their output
root explicitly. `phase3a_compare --help` and `phase3a_ensembles --help`
document the main interfaces. This describes the preserved Phase 3A handoff.

## Phase 3B: adaptive covariance search

The [Phase 3B report](results/RESEARCH_REPORT_PHASE_3B.md) completes the resumed
task: **33 SPD configurations attempted, 32 sampled**, wider crossing-exponent
calibration, provisional-sampler controls, and independent held-out
metric-measure comparisons. The promising region is A3 and nearby asymmetric
negative correlations, with finite linked xi approximately .225–.25 and gamma
.85–1.01. General-covariance graphs remain provisional; no unique parameter
curve or common continuum law is established. The earlier I3 xi=.3 candidate
remains relevant to large-graph and dimension checks.

- [Actual covariance matrices and reliability](results/phase3b/covariance_registry.json).
- [Calibration and conventions](docs/PARAMETER_CALIBRATION_PHASE3B.md).
- [Provisional sampler](docs/PROVISIONAL_SAMPLER_PHASE3B.md) and
  [saved-data quality audit](results/phase3b/quality/README.md).
- [Adaptive statistics](docs/ADAPTIVE_PROTOCOL_PHASE3B.md) and
  [independent structure review](docs/STRUCTURE_PHASE3B.md).
- [Final numerical comparisons and figures](results/phase3b/final_analysis/),
  [completion checkpoint](results/phase3b/STATUS.md), and
  [preservation manifest](results/phase3b/final_manifest.json).

All **173 tests pass**. Reanalyze existing data into a fresh directory:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=.mplconfig .venv/bin/python -m experiments.phase3b_results --output results/reproduction/phase3b_final --bootstrap 64
```

Simulation scripts `phase3b_crossings`, `phase3b_graphs`, `phase3b_fields` and
`phase3b_search` expose their configuration through `--help`; raw headers and
stage plans record the executed arguments and source hashes. Existing
experiments and raw data are preserved. Phase 3C has not begun.

The [12 November completion inventory](results/phase3b/finalization_20261112/COMPLETION_INVENTORY.md)
and [finalization manifest](results/phase3b/finalization_20261112/final_manifest.json)
record the final resumed checks: no new simulations, 173 passing tests, and
the existing datasets preserved. The report includes all 33 covariance
attempts and their best-tested linked discovery parameters.

## Phase 3C: broad family search — completed

The [Phase 3C report](results/RESEARCH_REPORT_PHASE_3C.md) covers **121 distinct
SPD covariance attempts,117 successful graph cohorts**, adaptive coverage,
selected refinement and frozen independent graph/field confirmation. It finds
extended finite-resolution compatibility regions, especially negative
equicorrelations and an asymmetric extension near linked xi=.225,gamma=.850,
but identifies no unique covariance map or convincing continuum correspondence
family. The proposed whole birectangular level segment fails componentwise
confirmation. Sampler, finite-size and compact-ensemble limitations remain.

- [Complete matrices, reliability and parameter profiles](results/phase3c/tables_final/all_covariances.json).
- [Final figures and comparison tables](results/phase3c/tables_final/).
- [Status and computational inventory](results/phase3c/STATUS.md).
- [Mathematical conventions](docs/MATHEMATICAL_PROTOCOL_PHASE3C.md).
- [Sampler error budget](docs/SAMPLER_RELIABILITY_PHASE3C.md).
- [Field construction and coupling](docs/FIELDS_PHASE3C.md).
- [Family-analysis protocol](docs/FAMILY_PROTOCOL_PHASE3C.md).
- [Integrity audit](results/phase3c/final_integrity.json) and
  [source/artifact manifest](results/phase3c/final_manifest.json).

**214 tests pass.** Existing datasets,paths andGit exclusions are preserved;
the report sits outside the excluded phase-results directory. Reproduction
commands use saved data and new output directories. Raw fields,paths,graphs
and metric caches require a separate checksummed backup even when Git ignores
them. No Phase3D work has begun.
