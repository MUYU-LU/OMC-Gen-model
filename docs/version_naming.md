# Version Naming Guide

This project tried many variants. Use these names consistently in notes, branches, and run directories.

## Recommended Version Families

### `base_uncond`

Unconditional MatterGen crystal generation.

```text
Generates: atomic_numbers + pos + cell
Use for: le50/le100/le150/le200/le250/le300 scaling baseline
```

### `csp_composition`

MatterGen CSP mode with fixed atomic composition.

```text
Generates: pos + cell
Conditions on: atomic_numbers / target composition
Does not condition on: SMILES graph
```

### `molcsp_fixed_graph`

Molecule-conditioned CSP with fixed per-node molecular graph conditioning.

```text
Generates: pos + cell
Conditions on: expanded molecule graph with fixed crystal atom indices
Use for: first working molecule-conditioned baseline
Known issue: atom-index switching in full-prior sampling
```

### `molcsp_topoloss`

Fixed graph conditioner plus topology auxiliary losses.

```text
Adds:
  fixed bond length loss
  fixed nonbond hard-negative loss
  Huber bond loss

Use for: low-noise recovery and early full-prior topology improvement
```

### `molcsp_assignment_bond`

Adds assignment-aware positive bond matching.

```text
Adds:
  unlabelled assignment-aware bond loss
  pair uniqueness
  endpoint usage cap
  mid-noise time window

Current strongest practical family:
  assign002_midfixed05 epoch14
```

### `molcsp_assignment_negative`

Adds assignment-aware negative / anti-merge terms.

```text
Result:
  did not improve overall
  worsened Z=4

Status:
  negative result; do not use as default starting point
```

### `molcsp_molgate`

Time-gated molecule conditioner.

```text
Goal:
  weaken fixed molecular role conditioning at high noise

Status:
  conceptually valid, first implementation did not beat baseline
```

### `molcsp_heavy_only`

Heavy-atom-only molecular CSP.

```text
Goal:
  remove H complexity

Result:
  stable cell/basic geometry
  topology still limited
```

### `molcsp_set_attention`

Pure unlabelled set-attention molecule conditioner.

```text
Goal:
  remove fixed-index role conflict

Result:
  failed target topology in quick full-prior sampling
  keep only as negative result
```

### `latent_assignment_topology_v1`

Proposed next clean direction.

```text
Start from:
  molcsp_assignment_bond / assign002_midfixed05 epoch14

Keep:
  fixed topology as low-noise regularizer

Add:
  identity-biased assignment matching
  mid-noise latent assignment topology

Avoid:
  assignment-negative until assignment matching is cleaner
```

## Naming Rules

Use this pattern for future runs:

```text
<dataset>_<family>_<key-loss-or-condition>_<start>_<important-hparams>
```

Examples:

```text
le50_molcsp_assignment_bond_start_topoloss364_w001
le50_molcsp_assignment_negative_start_assign002_w0005
le50_molcsp_set_attention_start_b_epoch294
le50_latent_assignment_topology_v1_start_assign002_epoch14
```

Avoid opaque names such as:

```text
tmp
test
new
final
best
fix2
```

## Metric Names

Use these evaluation terms:

```text
RDKit graph-isomorphism pass:
  generated components match the target SMILES up to atom relabelling

fixed-edge match:
  generated atom indices match the original mapped bond edges

MolTopo@k:
  at least one of k samples passes target molecular topology

ValidCrystal@k:
  at least one of k samples passes basic cell/clash filters

Sol@k:
  requires experimental reference crystal and structure matching
```

