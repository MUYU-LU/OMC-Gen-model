import json
import numpy as np

x = np.load('datasets/cache/omc25_raw/train/num_atoms.npy')
vals, counts = np.unique(x, return_counts=True)
dist = {int(v): float(c / counts.sum()) for v, c in zip(vals, counts)}
print(json.dumps(dist, indent=2, sort_keys=True))
