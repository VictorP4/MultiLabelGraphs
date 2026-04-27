import numpy as np, pandas as pd
from scipy.spatial.distance import pdist

paper = pd.read_csv("data/Hyperspheres_10_10_0/labels.csv").values
ours  = pd.read_csv("data/synthetic/hyperspheres_10_20_10/labels.csv").values

for name, L in [("paper", paper), ("ours", ours)]:
    d = pdist(L, metric="hamming")
    print(f"{name}: l_mean={L.sum(axis=1).mean():.2f}  "
          f"hamming median={np.median(d):.3f}  "
          f"frac<=0.10={(d<=0.10).mean():.4f}  "
          f"frac<=0.05={(d<=0.05).mean():.4f}")