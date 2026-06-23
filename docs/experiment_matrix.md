# OMC25 MolCSP Experiment Matrix

Short version of the tested variants. See `experiment_log.md` for detailed notes.

| # | Variant | Purpose | Result |
|---|---|---|---|
| 1 | Base unconditional le50-300 | Test atom-count scaling | le50/le100 usable; le200+ much harder. |
| 2 | Loss `sum` to `mean` | Remove atom-count gradient scaling | Clear positive; keep mean-style normalization. |
| 3 | Cutoff / max-neighbor sweeps | Test graph truncation | Not the main bottleneck. |
| 4 | CSP warm-start | Fix composition, generate pos/cell | Necessary but not sufficient for molecule identity. |
| 5 | CSP + Z / z_prim | Control molecule count with scalar condition | Too weak/ambiguous after primitive reduction. |
| 6 | OE62/SMILES mapping | Attach molecule graph to crystals | Strict explicit-H mapping is the cleanest format. |
| 7 | GNN molecule conditioner | Inject target graph into denoiser | Helps topology; fixed-index conflicts remain. |
| 8 | Cell variance 0.10 vs 0.25 | Tune cell corruption | 0.25 is the current molecular-CSP default. |
| 9 | Fixed bond loss | Enforce mapped covalent bond lengths | Useful, especially low-noise. |
| 10 | Huber bond loss | Stabilize bond outliers | Better than plain MSE/MAE; keep. |
| 11 | Fixed nonbond repulsion | Prevent false covalent contacts | Useful but can conflict with atom switching. |
| 12 | Assignment-aware bond loss | Allow unlabelled atom assembly | Best positive direction; improves RDKit graph pass. |
| 13 | Assignment-negative | Push away assignment false bonds/merge | Did not improve overall; worsened Z=4. |
| 14 | Molecule-conditioner gate | Weaken fixed roles at high noise | First implementation did not beat baseline. |
| 15 | Sampling-time energy guidance | Repair density/cell/bonds without training | Diagnostic only; does not solve topology. |
| 16 | Partial t-start recovery | Inspect denoising from reference corruption | Useful diagnostic, not full-prior generation. |
| 17 | Heavy-only training | Remove H complexity | Stable cell/basic geometry; topology still limited. |
| 18 | Literature comparison | Position against PackFlow/OXtal/Clari/etc. | Supports graph-conditioned, assignment-aware direction. |
| 19 | RL/physics alignment | Consider PackFlow-style reward tuning | Future stage after topology pass improves. |
| 20 | Pure set-attention conditioner | Remove fixed graph scaffold | Failed target topology; 0/64 on first two targets. |

## Current Baseline

The best practical baseline remains the assignment-aware molecule-conditioned CSP family:

```text
assign002_midfixed05 epoch14
full-prior eval: ~59.8% overall RDKit graph-isomorphism + basic pass
Z=1/Z=2: reasonable
Z=4: weak
```

See `study_results_summary.md` for the verified raw-report numbers and caveats.

## Current Bottleneck

The main bottleneck is:

```text
full-prior unlabelled atom assignment
plus multi-copy molecular assembly
```

Cell validity is no longer the dominant problem for le50 MolCSP.
