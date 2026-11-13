from itertools import permutations
import json
from pathlib import Path
import numpy as np
from src.covariance_design_3c import broad_design, coverage_families


def test_permutation_quotient_and_shared_independent_anchor():
    for p in permutations((-.2,.1,.35)):
        assert coverage_families(p)==['generic_asymmetric']
    assert coverage_families((0,0,0))==['birectangular','equicorrelated']


def test_broad_design_preserves_prior_points_and_avoids_spd_boundary():
    old=json.loads(Path('results/phase3b/extended_design.json').read_text())['configurations']
    design=broad_design(old)
    assert len(design)==109
    assert [(s['id'],s['triple']) for s in design[:len(old)]]==[(s['id'],s['triple']) for s in old]
    assert len({tuple(s['triple']) for s in design})==len(design)
    counts={g:sum(g in s['coverage_families'] for s in design) for g in
            ['birectangular','equicorrelated','two_correlation','generic_asymmetric']}
    assert counts==dict(birectangular=20,equicorrelated=15,two_correlation=25,generic_asymmetric=50)
    for s in design:
        a,b,c=s['triple'];det=1+2*a*b*c-a*a-b*b-c*c
        assert det>0 and np.isclose(det,np.linalg.det(s['covariance']))
        assert s['condition']<=12.+1e-10 and s['eigenvalues'][0]>=.15-1e-10
