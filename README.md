# MatterGen OMC25 Molecular CSP Experiments

This repository is a cleaned research snapshot for OMC25 molecular crystal structure prediction experiments built on top of MatterGen.

The original MatterGen README is kept as `README_MATTERGEN_ORIGINAL.md`. This README only documents the OMC25 molecular-CSP additions and what is included in this GitHub-ready export.

## What This Adds

- OMC25/OE62 molecule mapping utilities for SMILES and RDKit graph records.
- Molecule-conditioned CSP data loading with explicit atom, bond, molecule-copy, and optional heavy-atom fields.
- Molecule graph conditioners for GemNet-based MatterGen denoisers.
- Topology auxiliary losses:
  - fixed bond-length loss
  - fixed nonbond hard-negative repulsion
  - assignment-aware bond loss
  - assignment-negative diagnostics
- Full-prior molecular topology evaluation scripts using RDKit graph-isomorphism checks.
- Experiment notes summarizing tested variants and observed failure modes.

## Not Included

Large or private artifacts are intentionally excluded:

- OMC25/OE62 raw datasets
- generated CSV/cache datasets
- checkpoints and model weights
- Lightning outputs
- generated CIF samples
- machine-specific launch logs

Use environment variables or local paths to point scripts at your own data/checkpoints.

## Key Files

- `mattergen/common/data/molecule_dataset.py`: molecule-mapped crystal dataset support.
- `mattergen/common/gemnet/molecule_encoder.py`: molecular graph and set-attention conditioners.
- `mattergen/common/loss.py`: molecule topology losses and assignment-aware loss variants.
- `mattergen/denoiser.py`: integration point for molecule conditioning.
- `scripts/`: reusable molecule graph preparation and topology evaluation scripts.
- `docs/experiment_log.md`: experiment history, outcomes, and next-step notes.
- `docs/version_naming.md`: canonical names for tried and proposed version families.
- `docs/github_upload_inventory.md`: what was exported and what was excluded.

## Minimal Usage Sketch

Install the project as a local editable package:

```bash
pip install -e .
```

Set project/data paths explicitly:

```bash
export OMC25_ROOT=/path/to/this/repo
export MOLCSP_PYTHON=$(which python)
export WARMSTART_MODEL_PATH=/path/to/checkpoint_or_run_dir
```

Build molecule graph records from an OE62-like metadata table:

```bash
python scripts/build_oe62_smiles_graphs.py \
  --input /path/to/df_62k.json \
  --output datasets/molecule_mapping/oe62_smiles_graphs.jsonl
```

Evaluate generated CIFs with RDKit graph-isomorphism checks:

```bash
python scripts/molcsp_eval/check_rdkit_topology.py
```

Most scripts were originally research utilities. Review paths and CLI arguments before running them on a new machine.

## Current Research Status

The current best practical baseline is the assignment-aware molecule-conditioned CSP line documented in `docs/experiment_log.md`. The main remaining bottleneck is not cell validity; it is full-prior unlabelled atom assignment and multi-copy molecular assembly, especially for `Z=4`.

The next clean direction recorded in the notes is `latent_assignment_topology_v1`: keep fixed topology as a low-noise regularizer, but make mid-noise topology matching more assignment-aware.
