# Raw JSON Evidence Bundle

This folder contains small JSON summaries copied from the remote experiment tree on
2026-06-23. It intentionally excludes CIFs, trajectories, checkpoints, datasets,
and full Lightning output directories.

The files in `raw_json/` are the direct evidence for the headline numbers in:

```text
docs/study_results_summary.md
docs/full_experiment_history_analysis.md
docs/canonical_results_table.md
```

## Included Evidence

| File | Purpose |
|---|---|
| `analysis_molcsp_failure_epoch364_vs_epoch294_overall.json` | Epoch364 topology-loss improvement over epoch294. |
| `analysis_molcsp_epoch364_iso_aligned_by_z_summary.json` | Iso-aligned by-Z analysis showing fixed-index mismatch. |
| `le50_molcsp_epoch294_fullprior_suite_summary.json` | Early MolCSP full-prior suite result. |
| `le50_molcsp_assignbond_epoch4_suite_summary.json` | Early assignment-bond result. |
| `le50_molcsp_assignmask_epoch4_suite_summary.json` | Assignment-selected-pair masking result. |
| `le50_molcsp_assignmask_epoch19_suite_summary.json` | Later assignment-mask result. |
| `le50_assign002_midfixed05_epoch14_10x512_suite_summary.json` | Current best 10-target, 512-sample baseline. |
| `le50_assign002_midfixed05_epoch14_10x512_failure_decomposition_suite_summary.json` | Failure decomposition for the current best baseline. |
| `le50_assignneg_epoch19_10x128_suite_summary.json` | Assignment-negative comparison. |
| `le50_assignneg_epoch19_10x128_failure_decomposition_suite_summary.json` | Failure decomposition for assignment-negative. |
| `le50_molgate08_epoch14_10x64_suite_summary.json` | Molecule-conditioner gate comparison. |
| `le50_heavyonly_epoch19_summary_full_prior.json` | Heavy-only basic geometry diagnostic. |
| `le50_heavyonly_epoch19_heavy_rdkit_summary_full_prior.json` | Heavy-only topology diagnostic. |
| `le50_setattn_epoch31_*_rdkit_connectivity_summary.json` | Set-attention topology failure on two checked targets. |
| `le50_setattn_epoch31_*_shard00_summary_full_prior.json` | Representative set-attention basic-geometry shard summaries. |

## Interpretation Notes

`pass_basic_rate` in `summary_full_prior.json` is a cell/basic geometry metric. It
is not a target-SMILES topology metric.

The main molecular topology metric is `rdkit_basic_pass_rate_mean_over_targets`
from suite summaries, or `rdkit_molecule_pass_rate` from per-target RDKit
connectivity summaries.

The older epoch294 result has a one-sample source discrepancy:

```text
suite_summary: 2015 / 5120 = 39.36%
failure-decomposition report old_pass: 2016 / 5120 = 39.38%
```

This does not change the conclusion.
