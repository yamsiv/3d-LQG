"""Small immutable helpers for the Phase 3C field-family campaign.

The old field, GMC, LFPP and parameter-conversion implementations are reused.
This module only manages physical identities, linked probes and checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_XIS=(0.,.10,.125,.15,.175,.20,.2125,.225,.2375,.25,.2625,.275,.2875,.30)


def seed_for(base,*keys):
    return int(np.random.SeedSequence([base,*keys]).generate_state(1,dtype=np.uint64)[0])


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_new(path,data):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+".partial")
    with temporary.open("x") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n")
    temporary.rename(path)


def write_npz_new(path,**arrays):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+".partial")
    with temporary.open("xb") as stream:np.savez_compressed(stream,**arrays)
    temporary.rename(path)


def field_seed(path):
    with np.load(path) as archive:
        if "seed" in archive:return int(archive["seed"])
        return int(json.loads(str(archive["header"]))["seed"])


def gamma_probes(conversion,*,include_bands=True):
    if not conversion["valid"]:raise ValueError("No real finite subcritical conversion")
    probes=[("central",float(conversion["gamma"]))]
    if conversion["xi"]>0 and include_bands:
        for name,value in zip(["low","high"],conversion["gamma_systematic"]):
            if value is not None and 0<float(value)<np.sqrt(6):
                probes.append((name,float(value)))
    return probes


def sampled_points(probabilities,uniforms):
    cumulative=np.cumsum(np.asarray(probabilities,float).ravel())
    # Floating-point summation can end a few ulps below one. Set only the
    # known exact endpoint; do not clip or discard duplicate measure samples.
    cumulative[-1]=1.
    uniforms=np.asarray(uniforms,float)
    if np.any((uniforms<0)|(uniforms>=1)):raise ValueError("Uniforms must lie in[0,1)")
    return np.searchsorted(cumulative,uniforms,side="right")


def compatible_metric(path,*,source_hash,n,xi,stencil,probes,points):
    """Read-only check before reuse; every root and finite convention must agree."""
    path=Path(path)
    if not path.exists():return False
    with np.load(path) as archive:
        header=json.loads(str(archive["header"]))
        if (header.get("source_field_sha256")!=source_hash or header["n"]!=n
            or header["xi"]!=xi or header["stencil"]!=stencil
            or header.get("boundary","torus")!="torus"):
            return False
        old=header["probes"]
        if len(old)!=len(probes):return False
        if any(a["band"]!=b[0] or not np.isclose(a["gamma"],b[1],rtol=0,atol=2e-14)
               for a,b in zip(old,probes)):
            return False
        return archive["points"].shape==np.asarray(points).shape and np.array_equal(archive["points"],points)
