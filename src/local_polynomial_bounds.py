"""Exact rational box bounds for the stored polynomial Ritz factor.

This is a kernel-envelope feasibility tool, not a validated excursion sampler.
It removes the very pessimistic global Bernstein range ratio from a local
bound. Geometry/eigenfunction approximation errors are separate.
"""
from fractions import Fraction
from itertools import product
from math import comb
import numpy as np


def polynomial_box_bound(mode,center,halfwidth):
    """Bound q on a coordinate box by an exact shifted-polynomial expansion.

    For d=center+halfwidth*u, |u_i|<=1, expand q=sum c_gamma*u**gamma.
    Then q<=sum |c_gamma|. Floats are interpreted as their exact binary rationals.
    Bounds do not certify the numerical angular eigenfunction error.
    """
    center=np.asarray(center,float);halfwidth=np.broadcast_to(halfwidth,(3,))
    if np.any(center<=halfwidth) or np.any(halfwidth<0):
        raise ValueError("A strictly positive coordinate box is required")
    x=[Fraction.from_float(float(v)) for v in center]
    radius=[Fraction.from_float(float(v)) for v in halfwidth]
    degree=mode.degree
    xp=[[v**p for p in range(degree+1)] for v in x]
    rp=[[v**p for p in range(degree+1)] for v in radius]
    shifted={}
    for beta,value in zip(mode.exponents,mode.coefficients):
        value=Fraction.from_float(float(value))
        for gamma in product(*(range(int(b)+1) for b in beta)):
            term=value
            for i in range(3):
                b=int(beta[i]);g=gamma[i]
                term*=comb(b,g)*xp[i][b-g]*rp[i][g]
            shifted[gamma]=shifted.get(gamma,Fraction(0))+term
    central=shifted[(0,0,0)]
    upper=sum((abs(v) for v in shifted.values()),Fraction(0))
    lower=central-sum((abs(v) for k,v in shifted.items() if k!=(0,0,0)),Fraction(0))
    if central<=0:raise ValueError("Polynomial is not positive at the supplied center")
    return dict(center_value=float(central),lower=float(lower),upper=float(upper),
                upper_ratio=float(upper/central),terms=len(shifted),
                scope="exact polynomial box algebra for stored coefficients and binary-float box; no global path-law bound")
