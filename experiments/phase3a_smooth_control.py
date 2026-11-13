"""Small paired non-log Gaussian control, without new field randomness.

Hypothesis: root-ball heterogeneity alone need not diagnose logarithmic
correlations. Keep the point variance, replace the correlation structure by
a fixed smooth low-mode field. This is a finite negative control, not GMC.
"""
from pathlib import Path
import json
import numpy as np
from src.log_field_3d import spectral_configuration
from src.gmc_3d import gmc_weights
from src.lfpp_3d import build_lfpp_graph,root_distances
from src.metric_measure_3a import weighted_root_cdfs
from src.provenance import environment_manifest
from experiments.phase3a_ensembles import seed_for,write_npz


def main():
    root=Path("results/phase3a/smooth_control");root.mkdir(exist_ok=True)
    if (root/"metadata.json").exists():raise FileExistsError(root)
    n=32;xi=.3;gamma=1.5
    spectrum,_,variance=spectral_configuration(n)
    k=np.fft.fftfreq(n)*n
    radius2=sum(a*a for a in np.meshgrid(k,k,k,indexing="ij"))
    mask=(radius2>0)&(radius2<=4.)
    low_variance=float(spectrum[mask].sum())
    (root/"metadata.json").write_text(json.dumps({"provenance":environment_manifest(),
        "hypothesis":"Can smooth random metric/measure heterogeneity mimic the exploratory improvement?",
        "samples":8,"n":n,"xi":xi,"gamma":gamma,"common_fields":"confirmation first8 h32",
        "low_mode_radius":2.,"rescale_factor":float(np.sqrt(variance/low_variance)),
        "point_variance":variance,"estimated_seconds":5,
        "qualification":"Smooth fixed-band Gaussian control; exponential measure is not a log-field continuum GMC approximation"},indent=2)+"\n")
    for i in range(8):
        with np.load(f"results/phase3a/confirmation/field_{i:03d}.npz") as a:h=a["h32"]
        smooth=np.fft.ifftn(np.fft.fftn(h)*mask).real*np.sqrt(variance/low_variance)
        mu=gmc_weights(smooth,gamma,variance)
        points=np.searchsorted(np.cumsum(mu.probabilities.ravel()),
            np.random.default_rng(seed_for(202611113201,i,150,77)).random(8),side="right")
        graph,_=build_lfpp_graph(smooth,xi,stencil="26",boundary="torus")
        d=root_distances(graph,points);data=weighted_root_cdfs(d,mu.probabilities.ravel())
        header={"realization":i,"sampling_n":n,"n":n,"stencil":"26","pairs":[[xi,gamma]],
            "boundary":"torus","control":"smooth_nonlog_fixed_variance"}
        write_npz(root/f"metric_{i:03d}_n32_smooth.npz",grid=data["grid"][None,:],
            cdfs=data["cdfs"][None,:,:].astype(np.float32),root_means=data["root_means"][None,:],
            matrices=d[:,points][None,:,:],bin_width=np.array([data["distance_bin_width"]]),
            smooth_field=smooth,header=json.dumps(header))
    print("Saved 8 paired smooth non-log controls, no new Gaussian fields.")


if __name__=="__main__":main()
