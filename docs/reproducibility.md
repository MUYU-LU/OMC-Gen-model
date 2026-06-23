# Reproducibility Notes

This repository is a cleaned code and documentation snapshot. It does not include
datasets, checkpoints, generated CIFs, or full remote launch directories.

Use this file as a command map for reproducing the reported workflow on a machine
that has the OMC25/OE62 data and MatterGen environment.

## Environment Variables

Set paths explicitly:

```bash
export OMC25_ROOT=/path/to/OMC-Gen-model
export OMC25_DATA=/path/to/omc25_or_prepared_csvs
export OE62_JSON=/path/to/df_62k.json
export MOLCSP_OUT=/path/to/output_dir
export MOLCSP_SAMPLE_DIR=/path/to/sample_dir
export WARMSTART_MODEL_PATH=/path/to/checkpoint.ckpt
```

Install locally:

```bash
cd "$OMC25_ROOT"
pip install -e .
```

## Data And Molecule Mapping

Build OE62 SMILES graph records:

```bash
python scripts/data_prep/build_oe62_smiles_graphs.py \
  --input "$OE62_JSON" \
  --output "$OMC25_DATA/molecule_mapping/oe62_smiles_graphs.jsonl"
```

Map OMC25 CIF fragments to explicit-H molecule graphs:

```bash
python scripts/data_prep/map_omc25_cif_fragments.py \
  --input-csv "$OMC25_DATA/omc25_le50/train.csv" \
  --molecule-graphs "$OMC25_DATA/molecule_mapping/oe62_smiles_graphs.jsonl" \
  --output "$OMC25_DATA/molecule_mapping/omc25_le50_molgraph.jsonl"
```

The preferred mapping mode in the notes is:

```text
rdkit_explicit_h_full_match
```

Hybrid mapping was useful for diagnosis but is not the clean training target.

## Training Families

The exact remote Hydra launch wrappers were machine-specific and are not included.
Use these as config-level targets:

| Family | Key switches |
|---|---|
| Base unconditional | generate `atomic_numbers + pos + cell` |
| CSP-lite | `include_atomic_numbers=false`, generate `pos + cell` |
| MolCSP fixed graph | strict molecule graph fields enabled, molecular conditioner enabled |
| Bond Huber | `bond_weight`, `relative_huber`, `detach_cell=true` |
| Assignment bond | assignment-aware bond loss enabled, mid-noise active window |
| Assign002 midfixed05 | assignment bond enabled, fixed topology downweighted to 0.5 in mid-noise |
| Assignment negative | assignment negative / anti-merge enabled, not current baseline |
| Molgate08 | molecule-conditioner high-noise gate enabled |
| Heavy-only | heavy atom graph/data mode |
| Set-attention | pure unlabelled set-attention conditioner |

## Sampling

Full-prior target sampling template:

```bash
python scripts/molcsp_sampling/sample_fixed_target_fullprior.py \
  --checkpoint "$WARMSTART_MODEL_PATH" \
  --material-id "$MOLCSP_MATERIAL_ID" \
  --out-dir "$MOLCSP_SAMPLE_DIR" \
  --num-samples 512 \
  --num-steps 1000
```

Recovery / t-start sampling is diagnostic only:

```bash
python scripts/molcsp_sampling/sample_fixed_target_recovery.py \
  --checkpoint "$WARMSTART_MODEL_PATH" \
  --material-id "$MOLCSP_MATERIAL_ID" \
  --t-start 0.3 \
  --num-steps 1000
```

Do not compare recovery metrics directly with full-prior metrics.

## Evaluation

Evaluate target topology:

```bash
python scripts/molcsp_eval/check_rdkit_topology.py \
  --sample-dir "$MOLCSP_SAMPLE_DIR"
```

Run the full-prior suite:

```bash
python scripts/molcsp_eval/run_fullprior_eval_suite.py \
  --checkpoint "$WARMSTART_MODEL_PATH" \
  --out-dir "$MOLCSP_SAMPLE_DIR" \
  --num-samples-per-target 512 \
  --num-steps 1000
```

Run failure decomposition:

```bash
python scripts/molcsp_eval/failure_decomposition.py \
  --sample-dir "$MOLCSP_SAMPLE_DIR"
```

## Metric Caveats

`pass_basic_rate` is a cell/basic geometry metric.

`rdkit_molecule_pass_rate` checks target molecule graph isomorphism for a single
target folder.

`rdkit_basic_pass_rate_mean_over_targets` is the main suite-level metric used in
the results summary.

`Sol@k` is not reported here because it requires an experimental reference
packing and structure matching, which was not part of the current full-prior
screening suite.
