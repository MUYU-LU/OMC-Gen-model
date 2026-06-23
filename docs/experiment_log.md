# OMC25 MatterGen Molecular CSP Experiment Log

Date: 2026-06-09

This note records the major hypotheses we tested, how each was tested, what happened, and what we should do next. The goal is to keep the project from repeatedly re-testing the same ideas.

## Current One-Line Status

We moved from unconditional crystal generation toward SMILES/molecule-conditioned CSP. The cell/packing geometry is now mostly stable for le50 molecular CSP, but full-prior generation still struggles with unlabelled atom assignment and multi-copy molecular assembly, especially `Z=4`.

Current best practical baseline:

```text
full-atom molecule-conditioned CSP
start/checkpoint family: assign002_midfixed05 epoch14
full-prior eval: about 59.8% overall RDKit graph-isomorphism + basic pass
Z=1: good
Z=2: moderate/good
Z=4: still weak
```

Recent failed direction:

```text
pure unlabelled set-attention molecule conditioner
start: bondhuber_lowt04_posonly epoch294
result: stable cell/basic geometry but 0/64 RDKit target pass on NUNLAJ and KEZCUP
lesson: removing fixed molecular topology from the conditioner makes target SMILES identity too weak
```

The next clean experiment should be:

```text
latent_assignment_topology_v1
start from old assign002_midfixed05 epoch14
add time-dependent identity bias to assignment matching
disable assignment-negative
keep fixed topology only as low-t regularizer
evaluate by graph-isomorphism, not fixed atom-index match
```

## Evaluation Language

Important distinction:

```text
fixed-edge match:
  generated atom index i must form the original fixed bond i-j.
  This is not the right main metric for full-prior unlabelled generation.

RDKit / graph-isomorphism pass:
  generated connected components must be graph-isomorphic to the target SMILES molecule.
  Atom labels may be remapped after generation.
  This is the correct main topology metric for the current goal.
```

For new molecules without known experimental crystal:

```text
Use:
  MolTopo@k
  ValidCrystal@k
  diversity of generated packings
  relaxability / MLIP energy after topology pass

Do not use:
  Sol@k

Reason:
  Sol@k needs a reference experimental crystal packing.
```

## Experiments Tried

### 1. Base / Unconditional MatterGen By Atom-Count Buckets

Purpose:

Test whether models trained on `le50`, `le100`, `le150`, `le200`, `le250`, `le300` can model larger organic molecular crystals.

How tried:

```text
Train unconditional/base MatterGen variants on OMC25 atom-count buckets.
Base model denoises:
  atomic_numbers
  pos
  cell
```

Main observations:

```text
le50 and le100 train/use relatively well.
le150 is still usable but pos/atomic losses are larger.
le200/le250/le300 become much harder, especially under original loss scaling and small batch.
cell loss remains easier than pos/atomic for smaller systems, but can degrade for larger systems.
```

Conclusion:

Larger molecular crystals are harder not just because of cell size, but because the denoiser must assemble larger noisy periodic graphs and more atom identities. This pushed us toward CSP/molecule-conditioned generation instead of unconditional generation.

Status:

Keep as baseline and scaling evidence, but not the main next step.

### 2. Loss Normalization: `reduce=sum` To `reduce=mean`

Purpose:

Test whether large systems look worse because loss/gradient scale grows with atom count.

How tried:

```text
Change loss reduction to mean / per-structure normalization.
Apply to larger buckets, especially le150/le200.
```

Main observations:

```text
Training became much more stable.
le150/le200 no longer looked as broken as with sum-like scaling.
This was one of the clearest positive changes.
```

Conclusion:

This is a confirmed fix. For variable-size organic molecular crystals, mean-style normalization is required for fair optimization across atom counts.

Status:

Keep.

### 3. Neighbor Budget / Cutoff Experiments

Purpose:

Test whether `max_neighbors=50` truncates important local environments, especially for larger systems.

How tried:

```text
Ran/attempted variants around:
  cutoff = 6
  max_neighbors = 300

Compared against default-like settings.
Also discussed clean/noisy neighbor statistics and periodic image duplication.
```

Main observations:

```text
Increasing max_neighbors did not directly solve le150+ quality.
It can increase graph/triplet cost and may include noisy periodic artifacts.
For cutoff=6, clean neighbor p99 was far below 300, so 300 is not always meaningfully different from a smaller cap in clean data.
```

Conclusion:

Neighbor truncation may contribute, but it is not the main current bottleneck. The main bottleneck shifted to molecular topology and unlabelled assembly.

Status:

Do not prioritize further neighbor sweeps until topology objective is cleaned up.

### 4. CSP Mode Warm-Start From Base-50

Purpose:

Move from unconditional generation to CSP-like generation: fixed atom list/composition, generate only `pos + cell`.

How tried:

```text
Use MatterGen CSP mode:
  include_atomic_numbers = false
  include_pos = true
  include_cell = true

Warm-start from base-50 checkpoint instead of resuming optimizer state.
```

Main observations:

```text
CSP mode is valid and works in the sense that atomic numbers/composition are fixed.
It reduces one large source of error: wrong atomic composition.
However, composition-only CSP does not know the target SMILES graph.
It cannot guarantee that the generated atoms form Z copies of the intended molecule.
```

Conclusion:

CSP mode is necessary but insufficient for molecular crystal generation. It must be combined with molecule graph conditioning or post-generation graph validation/remapping.

Status:

Keep as base formulation.

### 5. CSP + Z / `z_prim` Conditioning

Purpose:

Test whether explicit `Z` or primitive-cell formula-unit information can control the number of molecules in the generated cell.

How tried:

```text
Created/attempted CSP+Z variants.
Discussed raw Z vs z_prim after primitive/reduced cell processing.
Checked whether original dataset Z can be trusted after cell reduction.
```

Main observations:

```text
Naively using raw Z from join keys is unsafe because cells were primitive/reduced.
z_prim can be fractional in some conventions and does not always mean "integer molecule count" in a direct model-friendly way.
In sampling, changing z_prim did not clearly change generated molecule count.
```

Conclusion:

Z-like scalar conditioning alone is too weak. If molecule graph is given and the atomic list contains Z copies, then Z is implicitly represented by the number of graph copies. Controlling actual molecule count should be evaluated through connected components and graph isomorphism.

Status:

Do not rely on scalar Z as the main control mechanism.

### 6. OE62 / SMILES Mapping

Purpose:

Attach molecular SMILES/graphs to OMC25/CSD crystal samples.

How tried:

```text
Used OE62 metadata / df_62k.json and CSD refcodes.
Generated molecule graph fields:
  atomic numbers
  atom features
  bond list
  bond type
  molecule copy id
  molecule atom id
  crystal bond edges

Preferred strict mode:
  rdkit_explicit_h_full_match
```

Main observations:

```text
Some mappings fail due to H count / CIF preprocessing / asymmetric-unit vs reduced-cell issues.
Hybrid modes can create ambiguity; strict explicit-H full match is cleaner for training.
```

Conclusion:

Mapping quality is critical. Use the strict RDKit explicit-H full match for full-atom molecular graph conditioning.

Status:

Keep strict mapping as the main dataset format.

### 7. Molecule Graph Conditioning With GNN Encoder

Purpose:

Condition MatterGen on the target SMILES/molecule graph.

How tried:

```text
Added molecular graph fields to ChemGraph:
  mol_x
  mol_bond_edge_index
  mol_bond_attr
  mol_atom_id
  mol_copy_id
  mol_bond_d0

Added MolecularGINEEncoder / MolecularGraphConditioner.
Injected molecule embedding into GemNet atom embedding path.
```

Main observations:

```text
The model started to generate target-like molecules from full prior.
However, atom indices switch heavily.
Fixed-edge match is near zero even when RDKit graph-isomorphism pass is high.
```

Conclusion:

Molecule graph conditioning helps, but fixed per-node graph conditioning is not fully aligned with unlabelled full-prior generation. The model behaves like it generates unlabelled graph-isomorphic molecules, not fixed-index molecules.

Status:

Keep for now, but long-term should move toward target graph cross-attention / dynamic role assignment.

### 8. Cell Corruption Variance: `limit_var_scaling_constant`

Purpose:

Test whether cell corruption is too large or too small for molecular CSP.

How tried:

```text
Compared:
  cell limit_var_scaling_constant = 0.10
  cell limit_var_scaling_constant = 0.25

Also did recovery/full-prior checks.
```

Main observations:

```text
0.10 can make some recovery settings look stable but does not solve full-prior generation.
0.25 appears more natural for full-prior cell generation in the molecule-conditioned setting.
```

Conclusion:

Use `0.25` as the current default for molecule-conditioned CSP. Cell is not the main current bottleneck.

Status:

Keep `cell_var = 0.25`.

### 9. Fixed Bond Length Loss

Purpose:

Force target covalent bonds to have plausible lengths.

How tried:

```text
Added bond length loss on x0_hat using fixed mol_bond_edge_index.
Used:
  relative Huber
  bond_weight around 0.003
  detach_cell = true
```

Main observations:

```text
This substantially improved molecular topology for simpler cases.
detach_cell=true was important to avoid satisfying bond lengths by distorting the cell.
```

Conclusion:

Fixed bond loss is useful for low-noise/reference-corruption recovery and simple molecules, but it conflicts with full-prior unlabelled atom switching if used too strongly at mid/high noise.

Status:

Keep only as low-t regularizer.

### 10. Huber Bond Loss

Purpose:

Reduce sensitivity to rare huge bond-length errors.

How tried:

```text
Switched from simple relative MSE/MAE-like behavior to relative Huber.
Typical:
  huber_beta = 0.2
```

Main observations:

```text
Training became more stable.
Bond loss no longer overreacted as badly to outlier x0_hat predictions.
```

Conclusion:

Huber is a better default for topology auxiliary losses.

Status:

Keep.

### 11. Fixed Nonbond / Hard-Negative Repulsion

Purpose:

Prevent non-bonded pairs from collapsing into false covalent bonds.

How tried:

```text
Added hard-negative repulsion on x0_hat.
Use near nonbonded pairs under covalent-radius cutoff.
Typical:
  nonbond_weight = 0.001
  hard_negatives_per_bond = 4
  cutoff_scale = 0.9
  detach_cell = true
```

Main observations:

```text
Helps prevent local false bonds.
But under unlabelled full-prior generation, fixed nonbond can fight assignment-selected bonds if atom labels switch.
Masking assignment-selected pairs from fixed nonbond is needed.
```

Conclusion:

Useful but only if it is downweighted in mid-t and does not repel current assignment-selected pairs.

Status:

Keep as low-t / masked regularizer.

### 12. Assignment-Aware Bond Loss

Purpose:

Allow current atom cloud to form the target molecule without requiring original atom indices.

How tried:

```text
Implemented unlabelled assignment bond loss:
  group target bonds by element pair / bond type
  candidate generated atom pairs match element/bond group
  greedy match by relative distance error
  pair unique
  endpoint usage cap
  active around t = 0.30 to 0.80
```

Main observations:

```text
Z=1 and Z=2 improved.
Fixed-edge match stayed near zero, but RDKit graph-isomorphism pass improved.
This confirms the model is naturally doing unlabelled molecular assembly.
```

Representative eval:

```text
old assign002_midfixed05 epoch14, 10 targets x 512 x N=1000:
  overall RDKit+basic pass: 3061/5120 = 59.79%
  Z=1: 67.29%
  Z=2: 67.90%
  Z=4: 41.67%
```

Conclusion:

Assignment-aware bond loss is the right direction, but it needs time-dependent identity bias and cleaner handling of fixed topology conflicts.

Status:

Upgrade to latent assignment with identity bias.

### 13. Assignment-Negative / Anti-Merge Loss

Purpose:

Prevent assignment-selected components from forming extra false bonds or merging.

How tried:

```text
Added assignment_negative:
  weight = 0.0005
  active t = 0.30 to 0.75
  intra/inter negative pairs
  detach_cell = true
```

Main observations:

```text
Training val improved slightly inside the run.
But full-prior sampling did not improve overall.
Z=4 got worse.
```

Representative eval:

```text
assign-negative epoch19, 10 targets x 128 x N=1000:
  overall mean over targets: 59.77%
  Z=1: 69.73%
  Z=2: 71.88%
  Z=4: 34.38%

Compared to old large eval:
  overall: essentially unchanged
  Z=2: slightly better
  Z=4: worse
```

Conclusion:

This version of assignment-negative likely over-penalizes multi-copy assembly or adds unstable local constraints. Do not keep stacking losses on this checkpoint.

Status:

Disable for the next clean experiment.

### 14. Molecule Conditioner Gate (`molgate08`)

Purpose:

Reduce fixed molecule graph conditioning at high noise, because fixed atom role may conflict with unlabelled assembly.

How tried:

```text
Ran a time-gated molecule conditioner variant around molgate08.
```

Main observations:

```text
It did not outperform the baseline.
```

Conclusion:

The idea is conceptually valid, but this first implementation was not enough. The current easier next step is identity-biased latent assignment loss, not conditioner redesign.

Status:

Paused.

### 15. Training-Free Energy Guidance

Purpose:

Test whether sampling-time external gradients can repair density/cell/bond geometry.

How tried:

```text
Tried hand-designed energy guidance:
  density energy
  angle barrier
  near-singular cell barrier
  length anisotropy
  bond energy

Tried both:
  E(x_t)
  E(x0_hat) with scheduled guidance
```

Main observations:

```text
Guidance can affect density/cell.
It does not solve molecular topology assembly.
Cell/bond guidance can also create new artifacts if applied too strongly or at the wrong time.
```

Conclusion:

Good diagnostic tool, not the main training solution.

Status:

Do not prioritize until topology pass is stronger.

### 16. Partial Recovery / t-Start Sampling

Purpose:

Understand denoising behavior from different noise levels.

How tried:

```text
Started from reference-corrupted states at:
  t = 0.8
  t = 0.6
  t = 0.5
  t = 0.3
  t = 0.1

Exported trajectories in extxyz.
```

Main observations:

```text
Low-t recovery can preserve or recover nearby structure.
But this is not the same as full-prior generation.
At full-prior, there is no reference anchor, so atom assignment can switch.
```

Conclusion:

Recovery tests are useful diagnostics, but full-prior evaluation is the real target.

Status:

Use only as diagnostic.

### 17. Heavy-Only Training

Purpose:

Test whether removing hydrogens makes topology assembly easier.

How tried:

```text
Added heavy_only option in MoleculeMappedCrystalDataset.
Removed H from:
  atomic_numbers
  pos
  mol_x
  mol_atom_id
  mol_copy_id
  mol_bond_edge_index
  mol_bond_d0

Trained:
  le50_molcsp_heavyonly_ft_assign002_assignneg0005_midfixed05_warmstart_epoch14_b8_1gpu_2204
  start from full-atom assign002_midfixed05 epoch14
```

Training result:

```text
best checkpoint:
  epoch=19-loss_val=0.05.ckpt

best val:
  0.05408

full-atom comparable assign-negative val:
  0.03315
```

Quick sample result:

```text
full-prior quick val20, N=1000:
  valid_cell_rate = 100%
  no_clash_rate = 100%
  pass_basic_rate = 100%
  volume_ratio_median = 1.010
  density_ratio_median = 0.990

heavy-atom RDKit graph check:
  heavy_molecule_pass = 8/20 = 40%
```

Conclusion:

Heavy-only makes cell/basic geometry very stable but does not automatically solve topology, especially for `Z>1`.

Status:

Not the primary next step, but useful as a secondary baseline.

### 18. Literature Comparison

Purpose:

Position our results relative to current molecular crystal generation literature.

Papers inspected:

```text
PackFlow
MolCrystalFlow
OXtal
OrgFlow
Clari / Fast Organic CSP
Genarris3
```

Main observations:

```text
Rigid-body/template methods guarantee molecular identity because they generate packing of a given molecule/conformer.
Full-atom methods use molecule graph conditioning but still need topology/validity filtering.
Sol@k requires experimental crystal reference.
For new molecules, reference-free topology/validity metrics are needed.
PackFlow uses RL/physics alignment after pretraining, not as the first training stage.
```

Conclusion:

Our current problem is consistent with full-atom molecule-conditioned generators: topology must be explicitly evaluated and often filtered/remapped. RL/physics alignment should come after topology pass is stronger.

Status:

Use literature for framing, not immediate code changes.

### 19. Reinforcement Learning / Physics Alignment

Purpose:

Consider whether PackFlow-style reinforcement learning can improve generated crystals.

How considered:

```text
PackFlow uses physics alignment:
  sample K packings for same molecule/unit-cell context
  score by MLIP heavy-atom energy and force statistic
  convert to group-relative advantages
  update policy with KL to reference model
```

Main observations:

```text
This is possible in principle for our diffusion model, but implementation is more complex.
The current model still has topology failures; MLIP reward could reward low-energy but wrong-molecule structures unless topology pass is enforced first.
```

Conclusion:

Do not do full RL yet. First improve topology pass. A cheaper intermediate option is reward reranking + self-training after topology filtering.

Status:

Future stage.

### 20. Pure Set-Attention Molecule Conditioner From Clean B Checkpoint

Purpose:

Test whether replacing the fixed-index molecular GNN conditioner with a more unlabelled conditioner can reduce the conflict between fixed atom roles and full-prior atom switching.

How tried:

```text
Start checkpoint:
  B = le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3
  checkpoint: epoch=294-loss_val=0.04.ckpt

Code change:
  added MolecularSetAttentionConditioner
  target molecule graph roles are encoded by GINE
  each current crystal atom attends to same-element molecule roles in the same crystal
  GemNet integration unchanged: node_condition is added to atom hidden state

Training:
  output: le50_molcsp_setattn_from_b_epoch294_b4_8gpu_2204
  GPUs: 2204 cards 0-7
  batch_size.train = 4 per GPU
  lr = 1e-5
  max_epochs = 40
  cell_var = 0.25
  reduce = mean
  kept only weak low-t fixed bondhuber:
    bond_weight = 0.001
    bond_time_gate_center = 0.4
    bond_detach_cell = true
  no assignment bond
  no fixed nonbond
  no assignment-negative
```

Training result:

```text
Training was stable and completed 40 epochs.
Best checkpoint:
  <PROJECT_ROOT>/outputs/singlerun/2026-06-09/
    le50_molcsp_setattn_from_b_epoch294_b4_8gpu_2204/
    checkpoints/epoch=31-loss_val=0.05.ckpt

Final train metrics around epoch39:
  loss_train ~= 0.048
  pos_train ~= 0.347
  cell_train ~= 0.0132
  bond_lengths_train ~= 0.00131

Validation stayed around loss_val = 0.05, not better than B checkpoint's 0.04.
```

Full-prior sampling check:

```text
Sampling run:
  <PROJECT_ROOT>/samples/
    le50_setattn_epoch31_fullprior_eval_10x64_n1000_2204gpu01234567

Settings:
  checkpoint = epoch31
  N = 1000 reverse steps
  64 samples per target
  8 GPUs

Stopped after first two targets because both failed target topology:

NUNLAJ Z=1:
  valid_cell_rate = 1.0
  no_clash_rate = 1.0
  pass_basic_rate = 1.0
  rdkit_component_count_match_rate = 0.765625
  rdkit_molecule_pass_rate = 0.0
  rdkit_basic_pass_count = 0 / 64

KEZCUP Z=1:
  valid_cell_rate = 0.984375
  no_clash_rate = 1.0
  pass_basic_rate = 0.984375
  rdkit_component_count_match_rate = 0.359375
  rdkit_molecule_pass_rate = 0.0
  rdkit_basic_pass_count = 0 / 64
```

Interpretation:

```text
Pure set-attention preserves cell/basic geometry but loses the target SMILES topology.
The fixed-index GNN conditioner is imperfect, but it supplies a strong molecular topology scaffold.
Removing that scaffold makes the molecular identity signal too weak.

This failure does not mean unlabelled generation is wrong.
It means unlabelled conditioning must still carry bond topology strongly, e.g. via:
  fixed topology as low-t regularizer
  assignment-aware topology loss
  dynamic remapping / target graph cross-attention with bond-aware bias

Do not continue this pure set-attention line without adding explicit topology constraints.
```

Conclusion:

Pure role-set attention is over-relaxed for this task. The next useful path should keep fixed molecular topology somewhere in the model/loss while making the topology objective more assignment-aware.

Status:

Failed. Do not use `epoch=31-loss_val=0.05.ckpt` as a main checkpoint.

## Best And Failed Checkpoints To Remember

Useful old baseline:

```text
<PROJECT_ROOT>/outputs/singlerun/2026-06-07/
  le50_molcsp_ft_assign002_midfixed05_warmstart_epoch19_b4_1gpu_2204/
  checkpoints/epoch=14-loss_val=0.03.ckpt
```

Do not use as next start:

```text
<PROJECT_ROOT>/outputs/singlerun/2026-06-08/
  le50_molcsp_ft_assign002_assignneg0005_midfixed05_warmstart_epoch14_b4_1gpu_2204/
  checkpoints/epoch=19-loss_val=0.03.ckpt
```

Reason:

```text
assignment-negative epoch19 did not improve overall and reduced Z=4 pass.
```

Heavy-only reference:

```text
<PROJECT_ROOT>/outputs/singlerun/2026-06-08/
  le50_molcsp_heavyonly_ft_assign002_assignneg0005_midfixed05_warmstart_epoch14_b8_1gpu_2204/
  checkpoints/epoch=19-loss_val=0.05.ckpt
```

Failed set-attention reference:

```text
<PROJECT_ROOT>/outputs/singlerun/2026-06-09/
  le50_molcsp_setattn_from_b_epoch294_b4_8gpu_2204/
  checkpoints/epoch=31-loss_val=0.05.ckpt
```

Reason:

```text
stable training and valid cells, but NUNLAJ and KEZCUP full-prior RDKit target pass were both 0/64.
```

## Current Main Hypothesis

The central bottleneck is no longer cell geometry. It is:

```text
full-prior unlabelled atom assignment
plus multi-copy molecular assembly
```

Training reference corruption has a natural atom identity:

```text
clean x0
small noise
x_t

atom i is still near atom i
```

Full-prior sampling does not:

```text
random atom cloud
denoise
atoms form graph components

atom index can switch
```

Therefore the correct objective should treat assignment as latent:

```text
L_topo(x) = min_A L_topo(x, A)
```

where `A` maps generated atom slots to target molecule atom roles.

## Next Experiment Specification

Name:

```text
latent_assignment_topology_v1
```

Start:

```text
old assign002_midfixed05 epoch14
```

Loss:

```text
diffusion pos/cell
fixed bond loss as low-t regularizer
fixed nonbond loss as low-t regularizer
latent assignment bond loss active in mid-t
no assignment-negative
```

Key implementation:

```text
In _select_assignment_bond_pairs_for_sample:
  cost = distance_cost + lambda_id(t) * identity_cost

identity_cost:
  0.0 if candidate pair is original fixed target pair
  0.5 if candidate pair shares one endpoint with original pair
  1.0 if fully switched

lambda_id(t):
  large at low t
  small at mid/high t
```

Initial parameters:

```text
lr = 1e-5
batch_size.train = 4
batch_size.val = 4
max_epochs = 20
cell_var = 0.25
reduce = mean
bond_weight = 0.003
nonbond_weight = 0.001
assignment_bond_weight = 0.001
assignment active t = 0.30 to 0.80
fixed_topology_mid_t_scale = 0.5
detach_cell = true
new optimizer
```

Required logs:

```text
assignment_bond_identity_match_rate
assignment_bond_partial_identity_rate
assignment_bond_identity_bias
assignment_bond_nonfixed_pair_rate
assignment_bond_coverage
assignment_bond_unique_atom_ratio
assignment_bond_max_endpoint_usage
assignment_bond_distance_mae
nonbond_assignment_pair_in_fixed_nonbond_rate
```

Evaluation:

```text
10 targets x 128 samples x N=1000
RDKit graph-isomorphism pass
failure decomposition

Primary success criterion:
  Z=4 pass recovers/improves without hurting Z=1/Z=2 badly.
```

## Things Not To Prioritize Right Now

```text
More max_neighbor sweeps
More assignment-negative variants
More cell energy guidance
Scalar Z-only conditioning
Heavy-only as the main route
Pure set-attention conditioner without explicit topology constraints
Full RL / MLIP physics alignment
```

These may become useful later, but they are not the current bottleneck.
