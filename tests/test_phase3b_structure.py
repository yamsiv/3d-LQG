import numpy as np
import pytest

from experiments.phase3b_structure import _growth, _sensitivity


def test_growth_bootstrap_keeps_physical_paths_paired():
    small,large=[],[]
    for index,factor in enumerate([1.,2.,3.,8.]):
        for n,records in [(512,small),(2048,large)]:
            records.append({'root_means':np.array([1.,1.5])*factor*n**.25,
                            'header':{'physical_path_id':str(index),'id':'C0','seed':index,'n':n}})
    fit=_growth(small,large,np.random.default_rng(1),64)
    assert fit['beta_finite']==pytest.approx(.25)
    np.testing.assert_allclose(fit['mc95'],.25)
    large[0]['header']['physical_path_id']='another path'
    with pytest.raises(ValueError,match='identical physical'):
        _growth(small,large,np.random.default_rng(1),64)


def test_weight_sensitivity_uses_robust_probe_criterion():
    rows=[]
    # C0 has the lowest central score but an adverse uncertainty probe.
    # C1 is the winner for the declared max-over-probes criterion.
    for identifier,central,high in [('C0',.01,.8),('C1',.15,.2)]:
        for band,score in [('central',central),('high',high)]:
            rows.append({'id':identifier,'xi':.2,'gamma':.9,'band':band,
                         'eligible_primary':band=='central','components':{'pair':score}})
    fit=_sensitivity({'rows':rows,'unit_mass_rows':rows})
    for ranking in fit['rankings'].values():
        assert ranking[0]['id']=='C1'
        assert ranking[0]['score']==pytest.approx(.2)
