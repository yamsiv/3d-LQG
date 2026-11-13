"""Checkpointed coupled-resolution site-sum crossings for Phase 3B.

Each JSON checkpoint is one field, grid, xi and orientation. Fields share
Fourier Gaussians across the *entire requested size tuple*. Reproduction needs
the tuple, seed and frozen source, not the seed alone. No torus path shortcuts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from src.log_field_3d import sample_coupled_fields
from src.lfpp_3d import build_lfpp_graph, cube_crossing
from src.provenance import environment_manifest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"fields": args.fields, "sizes": sorted(set(args.sizes)),
              "xis": sorted(set(args.xis)), "seed": args.seed,
              "regularization": args.regularization,
              "infrared_remove_radius": args.infrared_remove_radius,
              "axes": args.axes, "cutoff_fraction": 1/3,
              "metric": "CHZ physical site sum; endpoints included; open cube",
              "rng": "numpy PCG64; field_seed=seed+field_index",
              "save_fields": args.save_fields}
    if (output/"config.json").exists():
        if json.loads((output/"config.json").read_text()) != config:
            raise ValueError("Existing experiment config differs; use a new output directory")
    else:
        json_new(output/"config.json", config)
        snapshot = output/"source"
        snapshot.mkdir()
        paths = ["src/log_field_3d.py", "src/lfpp_3d.py", "src/provenance.py",
                 "experiments/phase3b_crossings.py"]
        for path in paths:
            shutil.copyfile(path, snapshot/Path(path).name)
        json_new(output/"provenance.json", {"environment": environment_manifest(),
                 "producer_hashes": {p: digest(p) for p in paths},
                 "source_paper_hashes": {p: digest(p) for p in
                  ["papers/HighDim_Exponents.pdf", "papers/Weak_ExpMetrics.pdf"]}})
    provenance = json.loads((output/"provenance.json").read_text())
    for path, old in provenance["producer_hashes"].items():
        if digest(path) != old:
            raise ValueError(f"Producer changed during experiment: {path}")
    started = time.perf_counter()
    for index in range(args.fields):
        field_seed = args.seed+index
        def checkpoint(n, xi, axis):
            return output/f"cross_f{index:03d}_n{n}_xi{xi:g}_a{axis}.json"
        todo = [(n, xi, axis) for n in config["sizes"] for xi in config["xis"]
                for axis in args.axes if not checkpoint(n, xi, axis).exists()]
        if not todo:
            continue
        field_started = time.perf_counter()
        fields = sample_coupled_fields(config["sizes"], np.random.default_rng(field_seed),
                                      regularization=args.regularization)
        field_records = []
        for n in config["sizes"]:
            field = fields[n]
            h = field.h
            if args.infrared_remove_radius:
                k = np.rint(np.fft.fftfreq(n)*n).astype(int)
                radius2 = sum(x*x for x in np.meshgrid(k,k,k,indexing="ij"))
                coefficients = np.fft.fftn(h)
                coefficients[radius2 <= args.infrared_remove_radius**2] = 0
                h = np.fft.ifftn(coefficients).real
            h_hash = hashlib.sha256(h.tobytes()).hexdigest()
            field_records.append({"n": n, "h_sha256": h_hash,
                                  "field_metadata": field.metadata,
                                  "infrared_remove_radius": args.infrared_remove_radius})
            field_path = output/f"field_f{index:03d}_n{n}.npz"
            if index < args.save_fields and not field_path.exists():
                with field_path.open("xb") as stream:
                    np.savez_compressed(stream, h=h, field_seed=field_seed,
                                        metadata=json.dumps(field_records[-1]))
            for xi in config["xis"]:
                axes = [axis for axis in args.axes if not checkpoint(n, xi, axis).exists()]
                if not axes:
                    continue
                tick = time.perf_counter()
                if xi == 0:
                    for axis in axes:
                        json_new(checkpoint(n, xi, axis), {
                            "field_index": index, "field_seed": field_seed, "n": n,
                            "xi": xi, "axis": axis, "site_crossing": 1.,
                            "h_sha256": h_hash, "seconds": 0.,
                            "method": "exact flat site-sum identity"})
                    continue
                graph, metadata = build_lfpp_graph(h, xi, stencil="axis", boundary="cube")
                weights = np.exp(xi*h)
                build_seconds = time.perf_counter()-tick
                for axis in axes:
                    tick = time.perf_counter()
                    value = cube_crossing(graph,n,axis=axis,site_weights=weights)
                    json_new(checkpoint(n, xi, axis), {
                        "field_index": index, "field_seed": field_seed, "n": n,
                        "xi": xi, "axis": axis, "site_crossing": value,
                        "h_sha256": h_hash, "build_seconds": build_seconds,
                        "crossing_seconds": time.perf_counter()-tick,
                        "csr_bytes": metadata["csr_bytes"],
                        "method": "open-cube axis edges + source/target endpoint half costs"})
                del graph, weights
        index_path = output/f"field_f{index:03d}_index.json"
        if not index_path.exists():
            json_new(index_path, {"field_index": index, "field_seed": field_seed,
                     "fields": field_records, "seconds": time.perf_counter()-field_started})
        print(json.dumps({"field":index,"field_seed":field_seed,
                          "seconds":time.perf_counter()-field_started,
                          "elapsed_seconds":time.perf_counter()-started,
                          "peak_rss_native":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
    summary = {"fields":args.fields,"checkpoint_count":len(list(output.glob("cross_*.json"))),
               "expected_checkpoints":args.fields*len(config["sizes"])*len(config["xis"])*len(args.axes),
               "run_seconds":time.perf_counter()-started}
    if summary["checkpoint_count"] != summary["expected_checkpoints"]:
        raise ValueError("Incomplete output")
    if not (output/"COMPLETE.json").exists():
        json_new(output/"COMPLETE.json",summary)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--fields",type=int,default=20)
    parser.add_argument("--sizes",type=int,nargs="+",default=[32,48,64,96,128])
    parser.add_argument("--xis",type=float,nargs="+",default=[0,.1,.15,.2,.225,.25,.275,.3,.325,.35,.375])
    parser.add_argument("--seed",type=int,default=202611113510)
    parser.add_argument("--regularization",choices=["sharp","heat"],default="sharp")
    parser.add_argument("--infrared-remove-radius",type=float,default=0.)
    parser.add_argument("--axes",type=int,nargs="+",default=[0])
    parser.add_argument("--save-fields",type=int,default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
