# Full Experiment History And Analysis

This document summarizes the project from the beginning as a scientific phase analysis. For the literal date-ordered artifact trajectory, see `chronological_run_log.md`.

For machine-readable raw summaries and curated comparable results, see:

```text
docs/results/raw_json/
docs/canonical_results_table.md
docs/all_evaluation_summaries.tsv
```

## Evidence Levels

Use these labels when interpreting results:

```text
Level A:
  verified from saved JSON summaries or analysis reports.

Level B:
  supported by saved logs or repeated interactive checks, but not a clean comparable benchmark.

Level C:
  qualitative conclusion from visual sampling / discussion / incomplete run.
```

Do not compare Level A and Level B/C as if they were identical benchmark numbers.

## Main Timeline

### 1. OMC25 Data Buckets And Base Crystal Generation

Version family:

```text
base_uncond
```

Goal:

```text
Train unconditional MatterGen models on atom-count buckets:
le50, le100, le150, le200, le250, le300.
```

Model target:

```text
generate atomic_numbers + pos + cell
```

What was learned:

```text
le50 and le100 are trainable and produce usable samples.
le150 is still usable but shows larger pos / atomic-number losses.
le200, le250, le300 become substantially harder.
The difficulty increases mainly in pos and atomic-number generation; cell is easier at small N but also degrades for larger buckets.
```

Evidence level:

```text
Level B/C.
```

Reason:

```text
These early runs mostly saved training logs and generated CIF folders, not clean RDKit/topology JSON benchmark summaries.
```

Important conclusion:

```text
Unconditional generation is not the right final formulation for molecular CSP.
For organic molecular crystals, the target molecule should be known; generating atomic numbers is unnecessary and adds a large failure mode.
```

### 2. Loss Normalization

Version family:

```text
loss_reduce_mean
```

Goal:

```text
Check whether larger systems look worse because losses/gradients scale with number of atoms.
```

Change:

```text
reduce=sum-like behavior -> mean / per-structure normalization
```

Result:

```text
Training became much more stable for le150, and mean-style scaling clarified the large-system comparison.
le200 still had runtime/NCCL instability and only partial sampling evidence, so do not treat it as a clean completed benchmark.
```

Evidence level:

```text
Level B.
```

Conclusion:

```text
Keep mean-style normalization for variable-size organic molecular crystals.
Do not use raw atom-sum losses to compare le50 vs le200/le300.
```

### 3. Neighbor Budget / Cutoff

Version family:

```text
neighbor_ablation
```

Variants:

```text
default max_neighbors=50
cutoff=6, max_neighbors=300
other le100/le150 retries
```

Goal:

```text
Test whether max_neighbors=50 truncates local environments and explains le150+ difficulty.
```

Result:

```text
Increasing max_neighbors did not directly fix le150+.
Larger neighbor caps increase graph/triplet cost and can introduce noisy periodic artifacts.
For cutoff=6, clean neighbor statistics did not justify always using 300.
```

Evidence level:

```text
Level B.
```

Conclusion:

```text
Neighbor truncation may matter, but it is not the main current bottleneck.
The project moved toward molecule-conditioned topology instead.
```

### 4. Composition-Only CSP

Version family:

```text
csp_composition
csp_zprim
csp_molfp
```

Goal:

```text
Switch from unconditional generation to CSP:
fixed atomic_numbers / composition, generate only pos + cell.
```

Result:

```text
CSP mode is valid and necessary.
It removes the atomic-number generation failure mode.
However, composition-only CSP does not know the target SMILES graph.
Scalar Z / z_prim conditioning was too weak and did not reliably control molecule count.
Fingerprint conditioning was an exploratory bridge but not sufficient for exact molecular graph control.
```

Evidence level:

```text
Level B/C.
```

Conclusion:

```text
CSP is the right generation type, but composition-only CSP is insufficient.
Need explicit molecule graph conditioning.
```

### 5. OE62 / SMILES / Molecule Mapping

Version family:

```text
strict_explicit_h_mapping
```

Goal:

```text
Attach molecule graph information to OMC25/CSD crystals.
```

Fields added:

```text
mol_x
mol_bond_edge_index
mol_bond_attr
mol_atom_id
mol_copy_id
mol_bond_d0
```

Important mapping decision:

```text
Use rdkit_explicit_h_full_match as the clean strict mode.
Avoid hybrid mapping unless needed for diagnostics, because it can introduce ambiguity.
```

Result:

```text
Mapping works for many samples but fails for some due to H count mismatch, reduced/primitive cell handling, asymmetric-unit issues, and CSD/OE62 inconsistencies.
```

Evidence level:

```text
Level B.
```

Conclusion:

```text
Mapping correctness is a core dependency.
Bad atom mapping directly corrupts fixed graph conditioning and bond losses.
```

### 6. First Molecule Graph Conditioner

Version family:

```text
molcsp_fixed_graph
```

Goal:

```text
Condition the CSP denoiser on target molecular graph.
```

Implementation:

```text
Molecular GINE encoder / MolecularGraphConditioner
inject molecular embedding into GemNet atom path
```

Result:

```text
This made the model start generating target-like molecular structures.
However, full-prior sampling produced heavy atom-index/copy switching.
Generated molecules can be correct up to graph isomorphism while fixed-edge match stays near zero.
```

Evidence level:

```text
Level A/B.
```

Key verified support:

```text
analysis_reports/molcsp_iso_aligned_epoch364_eval/iso_aligned_by_z_summary.json
```

Important numbers from `epoch364` iso-aligned analysis:

| Z | Pass Rate | Fixed-Index Up-To-Copy Auto Rate | Median Aligned Bond MAE |
|---:|---:|---:|---:|
| 1 | 60.79% | 1.29% | 0.0093 A |
| 2 | 61.98% | 0.00% | 0.0084 A |
| 4 | 27.93% | 0.00% | 0.0112 A |

Conclusion:

```text
The model is naturally doing unlabelled molecular assembly.
Fixed-edge match is useful as a diagnostic, not as the main success metric.
```

### 7. Cell Variance And Recovery Experiments

Version families:

```text
cell010
cell025
t_start_recovery
cell_fixed_recovery
training_free_guidance
```

Goal:

```text
Understand whether cell corruption / density prior / sampling guidance is the bottleneck.
```

Representative saved summaries:

```text
samples/le50_molcsp_cell010_best_tstart_parallel_val20_n200/summary.json
samples/le50_molcsp_cell025_best_after_resume_tstart_parallel_val20_n200/summary.json
samples/le50_molcsp_cell025_bondhuber_epoch54_quick_val20_n200/summary.json
```

Important caution:

```text
Many early summaries report pass_basic_rate, which is a geometric/cell/clash metric, not target molecular topology.
Do not interpret pass_basic_rate=1.0 as target SMILES success.
```

Result:

```text
0.25 cell variance became the preferred practical setting for full-prior molecular CSP.
Cell/basic validity can be made stable.
Energy guidance can change density/cell behavior but does not solve molecular topology.
Recovery from low/noisy reference states is not equivalent to full-prior generation.
```

Evidence level:

```text
Level A for saved geometric summaries.
Level B for interpretation.
```

Conclusion:

```text
Cell is not the main current bottleneck for le50 MolCSP.
Topology is.
```

### 8. Fixed Bond Loss And Huber Bond Loss

Version families:

```text
molcsp_topoloss
bond_huber
```

Goal:

```text
Force mapped molecular bonds to have plausible lengths.
```

Implementation:

```text
fixed mol_bond_edge_index
x0_hat bond length loss
relative Huber
detach_cell=true
```

Result:

```text
Fixed bond losses improved molecular topology compared with the earlier molecule-conditioned model.
Huber made the auxiliary loss more stable than raw squared errors.
```

Verified report:

```text
analysis_reports/molcsp_failure_decomposition_epoch364_vs_epoch294/overall.json
```

Numbers:

```text
old_pass = 2016 / 5120 = 39.38%
new_pass = 2626 / 5120 = 51.29%
delta = +11.91 percentage points
```

Evidence level:

```text
Level A.
```

Conclusion:

```text
Topology auxiliary losses help substantially.
But fixed-index topology losses conflict with full-prior atom/copy switching if used too strongly.
```

### 9. Fixed Nonbond Hard-Negative Repulsion

Version family:

```text
fixed_nonbond
assignmask
```

Goal:

```text
Prevent non-bonded fixed-index pairs from collapsing into false covalent contacts.
```

Definition:

```text
nonbond candidates = all atom pairs not in fixed mol_bond_edge_index
penalty applies only if PBC distance is below covalent-radius cutoff
top hard negatives are used
```

Result:

```text
Useful for preventing obvious false bonds.
Can conflict with unlabelled assignment when a switched but chemically correct pair is nonbond under the original mapping.
Masking assignment-selected pairs from fixed nonbond is necessary.
```

Verified JSON suite results:

| Variant | Samples | Targets | Mean RDKit Basic Pass |
|---|---:|---:|---:|
| `assignbond_epoch4` | 64/target | 10 | 53.75% |
| `assignmask_epoch4` | 64/target | 10 | 57.81% |
| `assignmask_epoch19` | 64/target | 10 | 57.97% |

Evidence level:

```text
Level A for the suite results.
```

Conclusion:

```text
Masking assignment-selected pairs helped relative to early assignment-bond.
But the stronger later baseline was assign002_midfixed05.
```

### 10. Assignment-Aware Bond Loss

Version family:

```text
molcsp_assignment_bond
assign002_midfixed05
```

Goal:

```text
Let the current atom cloud form the target molecule without requiring original atom indices.
```

Implementation:

```text
group target bonds by element/bond type
select candidate generated pairs by distance to target bond length
pair uniqueness
endpoint usage cap
active in mid-noise window
fixed topology downweighted in mid-noise
```

Best verified baseline:

```text
samples/le50_assign002_midfixed05_epoch14_fullprior_eval_10x512_n1000_2204gpu01234567/suite_summary.json
```

Settings:

```text
targets = 10
samples per target = 512
denoise steps = 1000
checkpoint epoch = 14
```

Verified results:

| Group | Pass / Samples | Mean RDKit Basic Pass |
|---|---:|---:|
| Overall | 3061 / 5120 | 59.79% |
| Z=1 | 1378 / 2048 | 67.29% |
| Z=2 | 1043 / 1536 | 67.90% |
| Z=4 | 640 / 1536 | 41.67% |

Evidence level:

```text
Level A.
```

Conclusion:

```text
This is the best current baseline.
It confirms assignment-aware topology is the right direction.
Z=4 remains weak, so multi-copy assembly is still unresolved.
```

### 11. Assignment-Negative / Anti-Merge

Version family:

```text
molcsp_assignment_negative
assignneg
```

Goal:

```text
Add assignment-aware negative / anti-merge repulsion.
```

Verified report:

```text
samples/le50_assignneg_epoch19_fullprior_eval_10x128_n1000_2204gpu0234567/suite_summary.json
```

Results:

| Group | Pass / Samples | Mean RDKit Basic Pass |
|---|---:|---:|
| Overall | 765 / 1280 | 59.77% |
| Z=1 | 357 / 512 | 69.73% |
| Z=2 | 276 / 384 | 71.88% |
| Z=4 | 132 / 384 | 34.38% |

Evidence level:

```text
Level A.
```

Conclusion:

```text
Overall was not better than assign002_midfixed05.
Z=4 got worse.
Do not continue from assignment-negative as the main baseline.
```

### 12. Molecule Conditioner Gate

Version family:

```text
molcsp_molgate
molgate08
```

Goal:

```text
Reduce fixed molecular role conditioning at high noise.
```

Verified report:

```text
samples/le50_assign002_midfixed05_molgate08_epoch14_fullprior_eval_10x64_n1000_2204gpu4567/suite_summary.json
```

Results:

| Group | Pass / Samples | Mean RDKit Basic Pass |
|---|---:|---:|
| Overall | 371 / 640 | 57.97% |
| Z=1 | 179 / 256 | 69.92% |
| Z=2 | 122 / 192 | 63.54% |
| Z=4 | 70 / 192 | 36.46% |

Evidence level:

```text
Level A.
```

Conclusion:

```text
This first gate implementation did not beat the assignment-bond baseline.
The idea remains plausible but should not be the immediate next baseline.
```

### 13. Heavy-Only Training

Version family:

```text
molcsp_heavy_only
```

Goal:

```text
Remove hydrogen complexity and test whether heavy-atom topology is easier.
```

Verified reports:

```text
samples/le50_heavyonly_epoch19_quick_val20_n1000_2204gpu0/summary_full_prior.json
samples/le50_heavyonly_epoch19_quick_val20_n1000_2204gpu0/heavy_rdkit_summary_full_prior.json
```

Results:

```text
20 one-sample targets
valid_cell_rate = 100%
no_clash_rate = 100%
pass_basic_rate = 100%
heavy_molecule_pass_rate = 40%
```

Evidence level:

```text
Level A diagnostic, not a main benchmark.
```

Conclusion:

```text
Heavy-only stabilizes cell/basic geometry but does not solve target topology.
It is useful as a diagnostic baseline, not as the main route unless the task definition changes.
```

### 14. Pure Set-Attention Conditioner

Version family:

```text
molcsp_set_attention
```

Goal:

```text
Remove fixed-index molecular graph scaffold and use more unlabelled target-role attention.
```

Verified partial report:

```text
samples/le50_setattn_epoch31_fullprior_eval_10x64_n1000_2204gpu01234567
```

Results:

| Target | Z | Pass / 64 | Pass Rate |
|---|---:|---:|---:|
| NUNLAJ | 1 | 0 | 0.00% |
| KEZCUP | 1 | 0 | 0.00% |

Evidence level:

```text
Level A partial negative result.
```

Conclusion:

```text
Pure set-attention mostly keeps cell/basic geometry in the checked shards but loses target SMILES topology.
Do not continue pure set-attention without explicit topology constraints.
```

### 15. Literature / Flow Matching / RL

Version family:

```text
literature_analysis
```

Main references discussed:

```text
PackFlow
OXtal
OrgFlow
MolCrystalFlow
Clari / Fast Organic CSP with Unit Cell Flow Matching
```

Conclusion:

```text
The literature supports graph-conditioned molecular crystal generation, assignment/automorphism-aware objectives, fitted cell priors, and topology-aware validation.
RL/physics alignment should come after topology pass is stronger; otherwise reward can optimize low-energy wrong molecules.
```

Evidence level:

```text
Level C for our internal project, literature-backed for method direction.
```

## Selected JSON-Based Evaluation Summaries

The selected aggregated table is in:

```text
docs/all_evaluation_summaries.tsv
```

Important caveat:

```text
Some rows report only pass_basic_rate, which is geometric/basic validity, not target molecular topology.
Rows with rdkit_basic_pass_rate_mean_over_targets are the most comparable full-prior topology benchmarks.
The table is not exhaustive; use `docs/results/raw_json/` and `docs/canonical_results_table.md` for the verified headline evidence.
```

## Final Correct Analysis

The project moved through this sequence:

```text
unconditional generation
-> composition-only CSP
-> molecule graph-conditioned CSP
-> fixed topology losses
-> assignment-aware topology losses
```

The main verified improvement path is:

| Stage | Verified Mean RDKit Basic Pass |
|---|---:|
| Early MolCSP epoch294 | 39.36% |
| Topology losses epoch364 | 51.29% |
| Assignment-aware assign002_midfixed05 epoch14 | 59.79% |

Current best baseline:

```text
molcsp_assignment_bond / assign002_midfixed05 epoch14
```

Current main bottleneck:

```text
full-prior unlabelled atom assignment + multi-copy molecular assembly, especially Z=4
```

Next study should not restart from scratch. It should start from the verified assignment-aware baseline and make the assignment objective more self-consistent.

Recommended next experiment:

```text
latent_assignment_topology_v1
```

Recommended design:

```text
start from assign002_midfixed05 epoch14
keep fixed bond/nonbond as low-noise regularizers
add identity-biased assignment matching
make mid-noise topology loss assignment-aware
do not include assignment-negative at first
evaluate by RDKit graph-isomorphism pass, by-Z pass, and failure decomposition
```
