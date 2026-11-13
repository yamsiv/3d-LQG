"""Broad moderate-condition covariance coverage modulo coordinate relabeling."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.stats import qmc
from .covariance_design_3b import configuration, canonical_triple


def coverage_families(triple):
    x=np.asarray(triple,float);zero=np.isclose(x,0,atol=1e-12,rtol=0)
    families=[]
    if zero.sum()>=2:families.append('birectangular')
    if np.ptp(x)<1e-12:families.append('equicorrelated')
    if zero.sum()==1:families.append('two_correlation')
    if not zero.any() and np.ptp(x)>=1e-12:families.append('generic_asymmetric')
    return families


def broad_design(previous, *, seed=202611123011, condition_limit=12., eigenvalue_floor=.15):
    """109 total points including the old33, not109 newly simulated matrices.

    The one shared point in the20/15/25/50 group targets is I3. Maximin LHS
    selection is performed after SPD rejection/canonicalization; it does not
    imply a uniform probability measure on the covariance quotient.
    """
    specs=deepcopy(previous);seen={canonical_triple(s['triple']) for s in specs}
    if len(seen)!=len(specs):raise ValueError('Previous covariance duplicates')
    for s in specs:
        s['previous_family']=s['family'];s['phase3c_origin']='phase3b_reused_configuration'
        s['coverage_families']=coverage_families(s['triple'])
    def valid(x):
        try:s=configuration(x)
        except ValueError:return False
        return s['condition']<=condition_limit+1e-10 and s['eigenvalues'][0]>=eigenvalue_floor-1e-10
    def add(x,family):
        key=canonical_triple(x)
        if key in seen:return
        if not valid(key):raise ValueError(f'Invalid design point {key}')
        s=configuration(key,family=family,identifier=f'C{len(specs):03d}')
        s.update(phase3c_origin='new_structured_or_space_filling',coverage_families=coverage_families(key))
        specs.append(s);seen.add(key)
    def fill(candidates,family,target):
        pool=list(dict.fromkeys(canonical_triple(x) for x in candidates if valid(x)))
        pool=[x for x in pool if x not in seen]
        while sum(family in s['coverage_families'] for s in specs)<target:
            if not pool:raise ValueError(f'Empty candidate pool for {family}')
            references=np.array([s['triple'] for s in specs if family in s['coverage_families']])
            p=np.array(pool);distance=np.linalg.norm(p[:,None,:]-references[None,:,:],axis=2).min(axis=1)
            i=int(np.argmax(distance));add(pool.pop(i),family)
    fill([(rho,0,0) for rho in np.linspace(-.8,.8,321)],'birectangular',20)
    fill([(rho,rho,rho) for rho in np.linspace(-.35,.55,361)],'equicorrelated',15)
    two=qmc.scale(qmc.LatinHypercube(2,seed=seed).random(2048),[-.8,-.8],[.8,.8])
    fill([(a,b,0) for a,b in two if abs(a)>.025 and abs(b)>.025],'two_correlation',25)
    generic=qmc.scale(qmc.LatinHypercube(3,seed=seed+1).random(4096),[-.8]*3,[.8]*3)
    fill([x for x in generic if min(abs(x))>.02 and np.ptp(x)>.03],'generic_asymmetric',50)
    return specs
