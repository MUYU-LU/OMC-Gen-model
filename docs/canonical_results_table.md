# Canonical Results Table

This table is the public-facing result set. It separates comparable full-prior
topology evaluations from diagnostics that used different sample counts, partial
target sets, or geometry-only metrics.

Raw JSON evidence is stored in `docs/results/raw_json/`.

## Comparable Full-Prior Topology Runs

Primary metric:

```text
RDKit basic pass = pass_basic && RDKit target graph-isomorphism pass
```

| Version | Checkpoint / Source | Targets | Samples / Target | Steps | Overall | Z=1 | Z=2 | Z=4 | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Early MolCSP | epoch294 suite | 10 | 512 | 1000 | 39.36% | not extracted | not extracted | not extracted | superseded |
| Topology loss | epoch364 analysis report | 10 | 512 | 1000 | 51.29% | 60.79% | 61.98% | 27.93% | superseded by assignment-aware |
| Assignment bond | assignbond epoch4 | 10 | 64 | 1000 | 53.75% | not extracted | not extracted | not extracted | early positive |
| Assignment mask | assignmask epoch4 | 10 | 64 | 1000 | 57.81% | not extracted | not extracted | not extracted | positive but small eval |
| Assignment mask | assignmask epoch19 | 10 | 64 | 1000 | 57.97% | not extracted | not extracted | not extracted | positive but small eval |
| Assignment bond, mid fixed 0.5 | assign002_midfixed05 epoch14 | 10 | 512 | 1000 | 59.79% | 67.29% | 67.90% | 41.67% | current best baseline |
| Assignment negative | assignneg epoch19 | 10 | 128 | 1000 | 59.77% | 69.73% | 71.88% | 34.38% | do not continue as baseline |
| Molecule gate | molgate08 epoch14 | 10 | 64 | 1000 | 57.97% | 69.92% | 63.54% | 36.46% | did not beat baseline |

## Current Best Per-Target Results

Source:

```text
le50_assign002_midfixed05_epoch14_10x512_suite_summary.json
```

| Target | Z | Pass / 512 | Pass Rate |
|---|---:|---:|---:|
| NUNLAJ | 1 | 466 | 91.02% |
| KEZCUP | 1 | 488 | 95.31% |
| FUWRAQ | 1 | 372 | 72.66% |
| ANACEY | 1 | 52 | 10.16% |
| BARBOL | 2 | 440 | 85.94% |
| BEYPUR | 2 | 244 | 47.66% |
| DCLETH02 gener | 2 | 359 | 70.12% |
| AKOVOL01 | 4 | 203 | 39.65% |
| DCLETH02 press | 4 | 156 | 30.47% |
| RHODIN01 | 4 | 281 | 54.88% |

## Diagnostics

| Version | Evidence | Metric | Result | Interpretation |
|---|---|---|---:|---|
| Heavy-only | 20 one-sample targets | `pass_basic_rate` | 100% | heavy-only cell/basic geometry is stable |
| Heavy-only | 20 one-sample targets | `heavy_molecule_pass_rate` | 40% | heavy topology is not solved |
| Pure set-attention | NUNLAJ 64 samples | `rdkit_molecule_pass_rate` | 0% | target topology failed |
| Pure set-attention | KEZCUP 64 samples | `rdkit_molecule_pass_rate` | 0% | target topology failed |
| Pure set-attention | representative shard summaries | `pass_basic_rate` | mostly 100% | geometry can be valid while topology fails |

## Non-Comparable / Qualitative Phases

These phases produced useful information but do not have comparable saved JSON
benchmarks in the exported evidence bundle:

| Phase | Status |
|---|---|
| Base unconditional le50/le100/le150/le200/le250/le300 | training logs and qualitative sampling only; useful for scaling diagnosis |
| Composition-only CSP and CSP+Z/zprim | generated structures only; showed composition is insufficient for molecule identity |
| CSP+molfp unseen molecule sampling | generated structures only; useful diagnostic, no suite metric |
| Cutoff=6 / max_neighbors=300 ablations | training/log diagnostic; did not become the main bottleneck |
| Cell recovery / training-free guidance | diagnostic; not a full-prior topology solution |

Do not rank these against the full-prior RDKit topology suite.
