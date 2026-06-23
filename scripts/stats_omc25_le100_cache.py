import numpy as np
from pathlib import Path
for split in ["train", "val"]:
    x = np.load(Path("datasets/cache/omc25_le100_mattergen") / split / "num_atoms.npy")
    print(split, "n", len(x), "min", int(x.min()), "mean", float(x.mean()), "p50", float(np.percentile(x,50)), "p95", float(np.percentile(x,95)), "max", int(x.max()))
