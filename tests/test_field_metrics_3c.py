import numpy as np
from src.field_metrics_3c import control_field,metric_arrays
from experiments.phase3c_confirmation_fields import tasks


def test_smooth_projection_andtheoretical_variance():
    n=12;x,y,z=np.ogrid[:n,:n,:n]
    low=np.sin(2*np.pi*x/n)+np.zeros((n,n,n))
    h=low+.5*np.cos(6*np.pi*y/n)
    values,variance,metadata=control_field(h,'smooth')
    np.testing.assert_allclose(values,low*metadata['multiplicative_scale'],atol=4e-15)
    assert np.isclose(metadata['unscaled_low_variance']*metadata['multiplicative_scale']**2,variance)
    assert metadata['not_log_correlated']


def test_duplicate_gamma_rows_alias_without_duplicate_queries():
    n=5;x,y,z=np.ogrid[:n,:n,:n]
    h=np.sin(2*np.pi*x/n)+np.cos(2*np.pi*(y+z)/n)
    arrays,masses,_,counts=metric_arrays(h,.2,1.,[('central',.7),('low',.7),('high',1.1)],
        np.linspace(.02,.98,16),stencil='26')
    assert counts['new_unique_gamma_rows']==2
    for key in['grid','cdfs','root_means','matrices','points']:
        np.testing.assert_array_equal(arrays[key][0],arrays[key][1])
    assert masses[0]['gamma']==masses[1]['gamma']


def test_model_probes_named_globally_andinvalid_branches_flagged():
    config={'xis':[.225,.3],'diagnostic_model_controls':['log']}
    rows=tasks(config,['log'])
    first=rows[0];assert len(first[3])==8
    assert len({g for _,g in first[3]})==5
    assert 'all_Ninv_half'in rows[1][2]['invalid_models']
    assert 'model_all_Ninv_half'not in [label for label,_ in rows[1][3]]
    assert [label for label,_ in first[3]][:3]==['central','low','high']
