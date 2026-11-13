import numpy as np
from src.adaptive_search_3b import ScoreSurrogate


def test_gp_interpolates_smooth_scores_and_returns_nonnegative_acquisition_variance():
    rng=np.random.default_rng(731);x=rng.uniform(0,.4,(24,4));x[:,3]=.15+.15*x[:,3]/.4
    y=np.sum(x[:,:3]**2,axis=1)+(x[:,3]-.25)**2
    model=ScoreSurrogate(x,y,noise_floor=.002)
    p,s=model.predict(x)
    assert np.mean(abs(p-y))<.015
    assert np.all(s>=0)
