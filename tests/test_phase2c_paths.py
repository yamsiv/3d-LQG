import numpy as np
import pytest

from src.brownian_excursions import sample_independent_excursion
from src.mated_crt import cell_minima
from src.phase2c_paths import (thin_polygonal, ball_exit_laplace,
    coupled_gaussian_ball_walks, bridge_ball_missed_exit_bound)
from src.phase2c_paths import ball_exit_cdf,ball_exit_quantile,brownian_sphere_skeleton
from src.phase2c_paths import sample_wos_doob


def test_exact_minima_bound_all_nested_polygonal_minima():
    sample=sample_independent_excursion(32,3,np.random.default_rng(9821),oversample=64)
    exact=cell_minima(sample,32)
    previous=np.full_like(exact,np.inf)
    for q in [1,4,16,64]:
        coarse=thin_polygonal(sample,32,q)
        minima=cell_minima(coarse,32)
        assert np.all(exact <= minima+1e-14)
        assert np.all(minima <= previous+1e-14)
        assert coarse.segment_minima is None
        assert not coarse.metadata["cell_minima_exact"]
        previous=minima


def test_nonnested_grid_rejected():
    sample=sample_independent_excursion(16,3,np.random.default_rng(9),oversample=8)
    with pytest.raises(ValueError,match="nested"):
        thin_polygonal(sample,16,3)


def test_ball_laplace_transform_moments_and_scale():
    assert ball_exit_laplace(0)==1
    s=1e-4
    assert (1-ball_exit_laplace(s))/s == pytest.approx(1/3,rel=3e-5)
    second=(ball_exit_laplace(2*s)-2*ball_exit_laplace(s)+1)/s**2
    assert second == pytest.approx(7/45,rel=3e-4)
    assert ball_exit_laplace(2,radius=3)==ball_exit_laplace(18)


def test_gaussian_nested_skeletons_share_common_preexit_values():
    paths=coupled_gaussian_ball_walks(1,3,np.random.default_rng(7302),relative_steps=(.2,.1,.05))
    fine,tfine,_=paths[.05]
    for scale in [.2,.1]:
        coarse,tcoarse,_=paths[scale]
        assert np.linalg.norm(coarse[-1])==pytest.approx(1,abs=1e-12)
        assert np.all(np.diff(tcoarse)>0)
        for point,t in zip(coarse[:-1],tcoarse[:-1]):
            if t < tfine[-2]:
                index=int(round(t/(.05**2/3)))
                np.testing.assert_allclose(point,fine[index],atol=2e-14)
    # Polygonal crossings within the same coarse interval need not be
    # ordered, even though finer *grid* detection occurs no later.
    assert paths[.05][1][-1] <= paths[.2][1][-1]+.2**2/3


def test_missed_exit_bound_diagnoses_near_boundary_failure():
    assert bridge_ball_missed_exit_bound([0,0,0],[0,0,0],1,.001)<1e-200
    assert bridge_ball_missed_exit_bound([.999,0,0],[.999,0,0],1,.001)==1
    assert bridge_ball_missed_exit_bound([1,0,0],[0,0,0],1,.01)==1


def test_ball_cdf_quantile_including_tails_and_formula_switch():
    u=np.r_[np.geomspace(1e-13,.01,30),np.linspace(.01,.99,100),1-np.geomspace(1e-13,.01,30)]
    np.testing.assert_allclose(ball_exit_cdf(ball_exit_quantile(u)),u,atol=3e-7,rtol=0)
    assert abs(ball_exit_cdf(.3-1e-12)-ball_exit_cdf(.3+1e-12))<1e-10
    assert ball_exit_cdf(0)==0


def test_brownian_sphere_skeleton_stays_in_ball_and_bounds_terminal_error():
    points,times,info=brownian_sphere_skeleton(1.,3,np.random.default_rng(330),step_size=.1)
    assert np.all(np.linalg.norm(points[:-1],axis=1)<1)
    assert np.linalg.norm(points[-1])==pytest.approx(1,abs=1e-14)
    assert np.all(np.diff(times)>0)
    assert info["omitted_time_mean_bound"]<=2e-7/3
    assert info["max_subball_radius"]<=.1


@pytest.mark.parametrize("kind",["i3","a3"])
def test_new_wos_doob_pipeline_positive_and_preserves_graph_invariants(kind):
    from src.mated_crt import graph_from_excursion,validate_graph
    if kind=="a3":
        from src.weyl_a3 import A3AnalyticCone
        model=A3AnalyticCone()
    else:
        from src.doob_excursions import analytic_cone
        model=analytic_cone(np.eye(3))
    sample=sample_wos_doob(32,rng=np.random.default_rng(441),model=model,
        step_size=.03,relative_step=.2,origin_cutoff=.003,trace_balls=True)
    assert np.all(np.diff(sample.times)>0)
    assert np.all(sample.values>=0)
    assert sample.metadata["law"]=="biased"
    assert not sample.metadata["cell_minima_exact"]
    assert sample.metadata["leg_diagnostics"][0]["ball_trace"]
    checks=validate_graph(graph_from_excursion(sample,32))
    assert checks["connected"]
