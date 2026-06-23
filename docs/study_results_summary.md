# Study Results Summary

This note separates verified metrics from exploratory notes so the next study can continue from a clean baseline.

For the full chronological analysis from the beginning of the project, see `full_experiment_history_analysis.md`.

## Metric Definition

Unless stated otherwise, the main reported metric is:

```text
RDKit basic pass = pass_basic && RDKit graph-isomorphism molecule pass
```

This is a reference-free topology/validity metric. It is not `Sol@k`, because `Sol@k` requires an experimental reference packing and structure matching.

## Current Best Verified Baseline

### `molcsp_assignment_bond / assign002_midfixed05 epoch14`

Raw report:

```text
samples/le50_assign002_midfixed05_epoch14_fullprior_eval_10x512_n1000_2204gpu01234567/suite_summary.json
```

Sampling settings:

```text
targets: 10
samples per target: 512
denoise steps: 1000
checkpoint epoch: 14
```

Verified results:

| Group | Targets | Pass / Samples | Mean Pass Rate |
|---|---:|---:|---:|
| Overall | 10 | 3061 / 5120 | 59.79% |
| Z=1 | 4 | 1378 / 2048 | 67.29% |
| Z=2 | 3 | 1043 / 1536 | 67.90% |
| Z=4 | 3 | 640 / 1536 | 41.67% |

Per-target verified results:

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

Interpretation:

```text
This is the best practical baseline so far.
Cell/basic validity is mostly solved.
The remaining bottleneck is target topology for harder molecules and multi-copy assembly, especially Z=4.
```

## Verified Comparison Runs

### `molcsp_assignment_negative / assignneg epoch19`

Raw report:

```text
samples/le50_assignneg_epoch19_fullprior_eval_10x128_n1000_2204gpu0234567/suite_summary.json
```

Sampling settings:

```text
targets: 10
samples per target: 128
denoise steps: 1000
checkpoint epoch: 19
```

Verified results:

| Group | Targets | Pass / Samples | Mean Pass Rate |
|---|---:|---:|---:|
| Overall | 10 | 765 / 1280 | 59.77% |
| Z=1 | 4 | 357 / 512 | 69.73% |
| Z=2 | 3 | 276 / 384 | 71.88% |
| Z=4 | 3 | 132 / 384 | 34.38% |

Interpretation:

```text
Assignment-negative did not improve overall performance.
It slightly helped Z=1/Z=2 in this smaller eval but worsened Z=4.
Do not use this checkpoint as the default starting point.
```

### `molcsp_molgate / molgate08 epoch14`

Raw report:

```text
samples/le50_assign002_midfixed05_molgate08_epoch14_fullprior_eval_10x64_n1000_2204gpu4567/suite_summary.json
```

Sampling settings:

```text
targets: 10
samples per target: 64
denoise steps: 1000
checkpoint epoch: 14
```

Verified results:

| Group | Targets | Pass / Samples | Mean Pass Rate |
|---|---:|---:|---:|
| Overall | 10 | 371 / 640 | 57.97% |
| Z=1 | 4 | 179 / 256 | 69.92% |
| Z=2 | 3 | 122 / 192 | 63.54% |
| Z=4 | 3 | 70 / 192 | 36.46% |

Interpretation:

```text
The first molecule-conditioner gate did not beat the assignment-bond baseline.
The idea is still conceptually relevant, but this implementation is not the next baseline.
```

### `molcsp_set_attention / setattn epoch31`

Raw report:

```text
samples/le50_setattn_epoch31_fullprior_eval_10x64_n1000_2204gpu01234567
```

Sampling settings:

```text
targets evaluated: 2
samples per target: 64
denoise steps: 1000
checkpoint epoch: 31
```

Verified results:

| Target | Z | Pass / 64 | Pass Rate |
|---|---:|---:|---:|
| NUNLAJ | 1 | 0 | 0.00% |
| KEZCUP | 1 | 0 | 0.00% |

Interpretation:

```text
Pure set-attention preserves basic cell validity but loses target SMILES topology.
This is a clear negative result.
Do not continue pure set-attention without explicit topology constraints.
```

### `molcsp_heavy_only / heavy-only epoch19`

Raw reports:

```text
samples/le50_heavyonly_epoch19_quick_val20_n1000_2204gpu0/summary_full_prior.json
samples/le50_heavyonly_epoch19_quick_val20_n1000_2204gpu0/heavy_rdkit_summary_full_prior.json
```

Sampling settings:

```text
targets/samples: 20 one-sample targets
denoise steps: 1000
checkpoint epoch: 19
```

Verified results:

```text
valid_cell_rate = 100%
no_clash_rate = 100%
pass_basic_rate = 100%
heavy_molecule_pass_rate = 40%
```

Interpretation:

```text
Heavy-only makes cell/basic geometry stable.
It does not solve molecular topology, especially for multi-copy systems.
Because this was only 20 one-sample targets, treat it as a diagnostic, not a main benchmark.
```

## Older Verified Analysis Reports

### `epoch364` topology improvement over earlier baseline

Raw report:

```text
analysis_reports/molcsp_failure_decomposition_epoch364_vs_epoch294/overall.json
```

Verified result:

```text
samples = 5120
old_pass = 2016 / 5120 = 39.38%
new_pass = 2626 / 5120 = 51.29%
delta = +11.91 percentage points
```

Interpretation:

```text
Topology auxiliary losses gave a real improvement over the earlier molecule-conditioned baseline.
Later assignment-aware training improved further to ~59.8% on the 10-target 512-sample suite.
```

### Iso-aligned analysis of `epoch364`

Raw report:

```text
analysis_reports/molcsp_iso_aligned_epoch364_eval/iso_aligned_by_z_summary.json
```

Verified result:

| Group | Pass Rate | Fixed-index Up-To-Copy Auto Rate | Median Aligned Bond MAE |
|---|---:|---:|---:|
| Z=1 | 60.79% | 1.29% | 0.0093 A |
| Z=2 | 61.98% | 0.00% | 0.0084 A |
| Z=4 | 27.93% | 0.00% | 0.0112 A |

Interpretation:

```text
Successful samples are chemically correct after graph-isomorphism remapping, but they almost never preserve fixed atom indices.
This confirms that full-prior generation behaves like unlabelled molecular assembly.
Fixed-edge match is a diagnostic, not the main success metric.
```

## Correct Conclusions For Next Study

1. The current best baseline is `molcsp_assignment_bond / assign002_midfixed05 epoch14`, not `assignneg`, `molgate`, `setattn`, or `heavy-only`.

2. Cell geometry/basic validity is no longer the main limiting factor for le50 MolCSP.

3. The main failure mode is target molecular topology under full-prior unlabelled atom assignment, especially for `Z=4`.

4. The model can generate chemically correct target molecules, but successful samples often require graph-isomorphism remapping. Therefore fixed atom-index metrics must not be used as the primary success criterion.

5. Assignment-negative as implemented is not reliable; it may over-penalize multi-copy assembly.

6. Pure unlabelled set attention is too weak without explicit topology constraints.

7. Heavy-only is useful as a diagnostic baseline, but not enough to solve topology.

## Recommended Next Experiment

Use:

```text
latent_assignment_topology_v1
```

Start from:

```text
molcsp_assignment_bond / assign002_midfixed05 epoch14
```

Do:

```text
keep fixed bond/nonbond only as low-noise regularizers
add identity-biased assignment matching
make mid-noise topology loss assignment-aware
do not include assignment-negative initially
evaluate by RDKit graph-isomorphism pass, by-Z pass, and failure decomposition
```

Do not do next:

```text
do not continue pure set-attention
do not continue assignment-negative checkpoint
do not prioritize cell guidance
do not use fixed-edge match as the primary metric
```

## Verification Status

The numeric results in this file were re-read from raw JSON reports on 2026-06-23. Comparison caveat: sample counts differ across runs, so the strongest comparison is the 10-target directionality, not exact percent-level ranking unless rerun with identical sample counts.
