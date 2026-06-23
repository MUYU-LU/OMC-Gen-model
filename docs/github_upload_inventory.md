# GitHub Upload Inventory

This export was prepared from the working OMC25 MatterGen directory and cleaned for source-code upload.

## Included

- `mattergen/`: source package with molecular-CSP additions.
- `sampling_conf/`: small sampling config files.
- `scripts/data_prep/`: reusable molecule graph and dataset-preparation utilities.
- `scripts/molcsp_sampling/`: full-prior/recovery sampling utilities.
- `scripts/molcsp_eval/`: topology, connectivity, and failure-decomposition evaluation utilities.
- `README.md`: OMC25 molecular-CSP overview.
- `README_MATTERGEN_ORIGINAL.md`: original MatterGen README from the working copy.
- `LICENSE`, `pyproject.toml`: package metadata.
- `docs/experiment_log.md`: experiment matrix and current conclusions.
- `docs/experiment_matrix.md`: short table of tried variants.
- `docs/study_results_summary.md`: verified numerical results and next-study recommendation.
- `docs/version_naming.md`: canonical version-family names.

## Excluded

- `datasets/`
- `checkpoints/`
- `outputs/`
- `results/`
- `samples/`
- `logs/`
- raw OMC25/OE62 data files
- generated `.jsonl.gz` graph files
- model checkpoints and Lightning logs
- machine-specific launch shell scripts

## Main Modified Source Areas

- `mattergen/common/data/molecule_dataset.py`
  - molecule graph fields
  - strict explicit-H mapping support
  - heavy-only option

- `mattergen/common/gemnet/molecule_encoder.py`
  - molecular atom/bond encoders
  - GINE conditioner
  - set-attention conditioner experiment

- `mattergen/common/loss.py`
  - fixed bond-length loss
  - fixed nonbond repulsion
  - assignment-aware bond loss
  - assignment-negative loss experiment
  - logging metrics for topology conflicts

- `mattergen/denoiser.py`
  - molecular conditioner integration and gating hooks

- `mattergen/diffusion/config.py`, `mattergen/diffusion/run.py`
  - warm-start support used by CSP finetuning experiments

## Experiment Count

The cleaned experiment log records 20 major experiment directions. These are research directions, not 20 independent production-ready model variants.

The strongest retained baseline is the assignment-aware molecule-conditioned CSP line. Failed directions are still documented because they are important negative results.

## Upload Recommendation

Initialize Git in this export directory, review `git status`, then push only this cleaned directory.

Do not push the original working directory directly.
