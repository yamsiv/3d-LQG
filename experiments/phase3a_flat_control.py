"""Exact flat-torus Euclidean distances as a directional-stencil control."""
from pathlib import Path
import json
import numpy as np
from src.lfpp_3d import exact_flat_distances
from src.metric_measure_3a import weighted_root_cdfs
from src.provenance import environment_manifest
from experiments.phase3a_ensembles import seed_for,write_npz


def main():
    root=Path("results/phase3a/flat_exact");root.mkdir(exist_ok=True)
    meta=root/"metadata.json"
    if meta.exists():raise FileExistsError(meta)
    meta.write_text(json.dumps({"provenance":environment_manifest(),"n":48,"samples":24,
        "seed_base":202611113201,"geometry":"unit Euclidean three-torus",
        "accuracy":"exact distances between supplied grid points; uniform grid quadrature"},indent=2)+"\n")
    n=48;p=np.ones(n**3)/n**3
    for i in range(24):
        points=np.minimum((np.random.default_rng(seed_for(202611113201,i,0,77)).random(8)*n**3).astype(int),n**3-1)
        d=exact_flat_distances(n,points)
        data=weighted_root_cdfs(d,p)
        h={"realization":i,"n":n,"sampling_n":n,"stencil":"euclidean_exact",
            "pairs":[[0.,0.]],"boundary":"torus","discrete":False,"seed_base":202611113201}
        write_npz(root/f"metric_{i:03d}_n48_euclidean.npz",grid=data["grid"][None,:],
            cdfs=data["cdfs"][None,:,:].astype(np.float32),root_means=data["root_means"][None,:],
            matrices=d[:,points][None,:,:],bin_width=np.array([data["distance_bin_width"]]),header=json.dumps(h))
    print("Saved 24 exact flat Euclidean torus controls; no shortest-path simulation.")


if __name__=="__main__":main()
