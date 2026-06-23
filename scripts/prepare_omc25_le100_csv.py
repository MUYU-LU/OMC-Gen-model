import pandas as pd
from pathlib import Path
src = Path("datasets/omc25_final_by_crystal_key")
dst = Path("datasets/omc25_le100_mattergen")
dst.mkdir(parents=True, exist_ok=True)
for split in ["train", "val"]:
    inp = src / f"{split}_final_le100.csv"
    out = dst / f"{split}.csv"
    df = pd.read_csv(inp, usecols=["material_id", "cif"])
    df.to_csv(out, index=False)
    print(split, len(df), out)
