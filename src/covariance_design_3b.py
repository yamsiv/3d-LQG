"""Moderate-condition covariance designs modulo coordinate permutations."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc
from .covariance import from_correlations, to_angles


def canonical_triple(triple):
    """All six permutations of three edge labels are induced by S3 vertices."""
    values=np.asarray(triple,float)
    if values.shape!=(3,):raise ValueError("Three correlations required")
    return tuple(float(v) for v in np.sort(values))


def configuration(triple, *, family="asymmetric", identifier=None):
    key=canonical_triple(triple)
    C=from_correlations(*key);eigen=np.linalg.eigvalsh(C)
    return {"id":identifier,"triple":list(key),"covariance":C.tolist(),"family":family,
        "eigenvalues":eigen.tolist(),"condition":float(eigen[-1]/eigen[0]),
        "angles_radians":list(map(float,to_angles(C))),
        "coordinate_permutations_quotiented":True}


def design_covariances(*, seed=202611113611, generic_count=12, condition_limit=12.):
    """Anchored maximin Latin-hypercube candidates in the SPD quotient.

    LHS is generated before sorting/rejection. The final sorted quotient
    design is space filling, not a claim of uniform covariance prior.
    """
    specs=[];seen=set()
    def add(triple,family):
        key=canonical_triple(triple)
        if key in seen:return
        spec=configuration(key,family=family,identifier=f"C{len(specs):03d}")
        if spec['condition']>condition_limit or spec['eigenvalues'][0]<.15:return
        seen.add(key);specs.append(spec)
    for rho in [-.3,-.15,0.,.15,.3,.45]:add((rho,rho,rho),"equicorrelated")
    for rho in [-.7,-.5,-.25,.25,.5,.7]:add((rho,0,0),"birectangular")
    for rho in [-.5,-.3,.2,.4]:add((rho,0,rho),"symmetric_two_correlation")
    # Retain a generic cone already studied angularly in Phase2C.
    add((-.2,.1,.35),"asymmetric_control")
    candidates=[]
    for row in qmc.scale(qmc.LatinHypercube(3,seed=seed).random(256),[-.55]*3,[.65]*3):
        key=canonical_triple(row)
        try:spec=configuration(key)
        except ValueError:continue
        if spec['condition']<=condition_limit and spec['eigenvalues'][0]>=.15 and key not in seen:
            candidates.append(key)
    for _ in range(generic_count):
        coordinates=np.array([s['triple']for s in specs])
        distances=np.linalg.norm(np.asarray(candidates)[:,None,:]-coordinates[None,:,:],axis=2).min(axis=1)
        best=int(np.argmax(distances));key=candidates.pop(best)
        add(key,"latin_hypercube_asymmetric")
    return specs


def adaptive_candidates(existing, scores, *, count=4, seed=202611113612):
    """Covariance-only maximin proposals near several promising configurations.

    Root's joint C/xi GP supplies its own acquisition scores; this helper
    supplies a reproducible feasible local cloud and avoids redundant S3 copies.
    """
    rng=np.random.default_rng(seed);existing=np.asarray(existing,float)
    anchors=existing[np.argsort(scores)[:min(4,len(existing))]]
    candidates=[]
    for _ in range(400):
        row=canonical_triple(anchors[rng.integers(len(anchors))]+rng.normal(0,.10,3))
        try:spec=configuration(row,family="adaptive_local")
        except ValueError:continue
        if spec['condition']>12 or spec['eigenvalues'][0]<.15:continue
        if np.min(np.linalg.norm(existing-row,axis=1))<.035:continue
        candidates.append(spec)
    if not candidates:raise ValueError("No admissible local proposals")
    chosen=[]
    for _ in range(count):
        pool=np.asarray([c['triple']for c in candidates])
        reference=np.r_[anchors,np.array([s['triple']for s in chosen]).reshape(-1,3)]
        i=int(np.argmax(np.linalg.norm(pool[:,None,:]-reference[None,:,:],axis=2).min(axis=1)))
        chosen.append(candidates.pop(i))
    return chosen
