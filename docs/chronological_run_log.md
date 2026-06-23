# Chronological Run Log

This file is the date-ordered trajectory. It is intentionally different from
`full_experiment_history_analysis.md`, which groups experiments by scientific
idea.

Evidence levels:

```text
Level A = saved JSON metric or analysis report
Level B = saved logs / run artifacts / repeated checks
Level C = qualitative sampling or discussion only
```

## April 2026

| Date | Run Family | Artifact Examples | Result / Decision | Evidence |
|---|---|---|---|---|
| 2026-04-23 to 2026-04-24 | Base unconditional le50/le100/le150 | `outputs/singlerun/2026-04-23/*`, `outputs/singlerun/2026-04-24/*` | le50/le100 train and sample better than larger buckets; le150 starts showing larger pos/atom-type difficulty. | B/C |
| 2026-04-24 to 2026-04-26 | CSP warm-start and CSP scratch | `omc25_le50_csp_warmstart`, `omc25_le50_csp_scratch_1gpu_2203` | CSP mode is valid: fixed atom types/composition, generate pos/cell. Scratch controls existed but did not become the main path. | B/C |
| 2026-04-24 to 2026-04-26 | CSP+Z / zprim | `omc25_le50_csp_zprim_warmstart_from_base`, `omc25_le50_csp_zprim_scratch_1gpu_2204`, `results/zprim_effect_quick_z1/z2/z4` | Scalar Z/zprim conditioning was too weak and ambiguous after primitive/reduced-cell processing. | B/C |
| 2026-04-25 to 2026-05-02 | le200/le250/le300 data and base attempts | `omc25_le200`, `omc25_le250`, `omc25_le300` output/log folders | Large buckets were harder and not cleanly completed as comparable benchmarks. le250 terminated around epoch 117 and le300 around epoch 82 in available artifacts. | B |
| 2026-04-27 to 2026-04-28 | CSP+molfp | `omc25_le50_csp_molfp_*`, `results/omc25_le50_csp_molfp_unseen_mol27_c6h12n2o2_epoch149_400samples_0428` | Fingerprint conditioning was a bridge toward molecule conditioning, but not enough for exact target graph control. Generated structures exist, no comparable JSON benchmark found. | C |

## May 2026

| Date | Run Family | Artifact Examples | Result / Decision | Evidence |
|---|---|---|---|---|
| 2026-05-19 to 2026-05-20 | cutoff=6 / max_neighbors=300 | `le100_cutoff6_maxneigh300`, `le150_cutoff6_maxneigh300` | Increasing neighbor cap did not directly solve the large-system issue and increased graph/triplet cost. | B |
| 2026-05-25 to 2026-05-27 | loss reduce=sum to mean | `le150_default_maxneigh50_lossmean`, `le200_default_maxneigh50_lossmean_b1acc1_3gpu345` | Mean-style normalization helped, especially le150. le200 still had runtime/NCCL instability and only partial sampling evidence. | B |
| 2026-05-27 | molgraph smoke tests | `molgraph_smoke_val_as_train_gpu0`, `molgraph_smoke200_val_as_train_gpu0` | Verified the mapped molecule graph fields could enter training on small smoke runs. | B |
| 2026-05-27 onward | le50 MolCSP scratch and cell variance | `le50_molcsp_scratch_mean_cell010`, `le50_molcsp_scratch_mean_cell025` | Cell variance 0.25 became the preferred practical setting for full-prior MolCSP, but this is not a standalone benchmark conclusion. | B/C |

## June 2026

| Date | Run Family | Artifact Examples | Result / Decision | Evidence |
|---|---|---|---|---|
| 2026-06-01 to 2026-06-03 | recovery, cell fixed, training-free guidance | `cell010/cell025 tstart`, `strictN1000`, `cellfixed`, guidance sample folders | Recovery tests diagnosed denoising behavior, but reference recovery is not full-prior generation. Energy guidance can move density/cell/bond geometry but does not solve topology. | A/B diagnostic |
| 2026-06-03 | raw bond loss then Huber bond loss | `le50_molcsp_scratch_mean_cell025_bondw1e-3_8gpu01234567`, `bondhuber` folders | Fixed bond length loss helped; Huber stabilized the auxiliary topology loss. | B/A |
| 2026-06-04 | topology loss epoch364 | `analysis_reports/molcsp_failure_decomposition_epoch364_vs_epoch294/overall.json` | Improved full-prior topology from about 39.38% to 51.29%. | A |
| 2026-06-05 | assignment-bond smoke and masking | `assignbond_epoch4`, `assignmask_epoch4`, `assignmask_epoch19` | Assignment-aware positive bond matching and masking assignment-selected pairs from fixed nonbond improved over the earliest assignment-bond result. | A |
| 2026-06-07 | assign002_midfixed05 | `le50_assign002_midfixed05_epoch14_fullprior_eval_10x512_n1000_*` | Current best: 3061/5120 = 59.79%; Z=4 still weak at 41.67%. | A |
| 2026-06-08 | assignment-negative / anti-merge | `le50_assignneg_epoch19_fullprior_eval_10x128_n1000_*` | Not better overall and worse for Z=4; do not continue this as the baseline. | A |
| 2026-06-08 | heavy-only | `le50_heavyonly_epoch19_quick_val20_n1000_*` | Basic geometry 100%, heavy topology 40% on a 20-sample diagnostic. Useful diagnostic, not the main route. | A diagnostic |
| 2026-06-09 | pure set-attention | `le50_setattn_epoch31_fullprior_eval_10x64_n1000_*` | Partial 2-target eval had 0/64 target topology for both NUNLAJ and KEZCUP. Representative shard geometry was mostly valid, so the failure is target topology, not necessarily cell geometry. | A partial |

## Current State

Best baseline:

```text
molcsp_assignment_bond / assign002_midfixed05 epoch14
```

Main bottleneck:

```text
full-prior unlabelled atom assignment and multi-copy molecular assembly,
especially Z=4.
```

Next clean experiment:

```text
latent_assignment_topology_v1
```

Start from the clean assignment-aware baseline, not from `assignneg`.
