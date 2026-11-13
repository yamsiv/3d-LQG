"""Independent correction-model diagnostics for Task 2B.

Equation (45) concerns collapsed histogram scales, not mean distances. Its
reference-normalized form below sets a+b=1 because k(n_ref)=1 identically.
This explicitly stated restriction differs from fitting free a near one.
All intervals remain conditional on the correction ansatz and size window.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def fit_mean_model(sizes, means, errors, model="offset"):
    sizes, means, errors = map(lambda x: np.asarray(x, dtype=float), (sizes, means, errors))
    x = sizes/sizes.max()

    def profile(p, full=False):
        if model == "power":
            design = np.column_stack([x**p])
        elif model == "offset":
            design = np.column_stack([x**p, np.ones(len(x))])
        elif model == "offset_inverse_power":
            design = np.column_stack([x**p, np.ones(len(x)), x**(-p)])
        elif model.startswith("multiplicative_delta_"):
            delta = float(model.rsplit("_",1)[1])
            design = np.column_stack([x**p, x**(p-delta)])
        else:
            raise ValueError(model)
        coefficient = np.linalg.lstsq(design/errors[:,None], means/errors, rcond=None)[0]
        residual = (design@coefficient-means)/errors
        loss = float(residual@residual)
        return (loss,coefficient) if full else loss

    # Scan before local minimization: correction profiles can have several basins.
    grid = np.linspace(.03,.8,180)
    losses = np.array([profile(p) for p in grid])
    candidates = [0,len(grid)-1]+[i for i in range(1,len(grid)-1)
        if losses[i] <= losses[i-1] and losses[i] <= losses[i+1]]
    fits = [minimize_scalar(profile, bounds=(grid[max(i-1,0)],grid[min(i+1,len(grid)-1)]),
                            method="bounded") for i in candidates]
    best = min(fits,key=lambda f:f.fun)
    loss, coefficient = profile(best.x,True)
    k = len(coefficient)+1
    return dict(model=model,p=float(best.x),dimension=float(1/best.x),
                coefficients=coefficient.tolist(),chi2=loss,dof=len(x)-k,
                positive_leading_amplitude=bool(coefficient[0]>0),
                boundary=bool(best.x < .031 or best.x > .799),
                aicc=float(loss+2*k+2*k*(k+1)/(len(x)-k-1)) if len(x)>k+1 else None)


def fit_equation45(sizes, scales, covariance=None, *, delta_ratio=1., b_bound=.1):
    """Fit k=x^-p*(1-b+b*x^(-delta)), delta=delta_ratio*p.

    The deterministic reference scale is omitted from residuals. With supplied
    covariance, full GLS propagates the shared random reference histogram.
    A bounded small b is a modelling assumption; boundary hits are reported.
    """
    sizes, scales = np.asarray(sizes),np.asarray(scales)
    x = sizes[:-1]/sizes[-1]
    target = scales[:-1]
    if covariance is None:
        whitening=np.eye(len(x))
    else:
        eigenvalues,eigenvectors=np.linalg.eigh(np.asarray(covariance))
        floor=max(float(eigenvalues.max())*1e-7,1e-12)
        whitening=(eigenvectors/np.sqrt(np.maximum(eigenvalues,floor))).T

    def prediction(params):
        p,b=params
        return x**(-p)*(1-b+b*x**(-delta_ratio*p))

    def residual(params):
        return whitening@(prediction(params)-target)

    fits=[least_squares(residual,[p,b],bounds=([.03,-b_bound],[.8,b_bound]),max_nfev=500)
          for p in [.15,.23,.35] for b in [-.01,.01]]
    fit=min(fits,key=lambda f:float(f.fun@f.fun))
    p,b=fit.x
    return dict(model="Eq45_reference_normalized_a=1-b",p=float(p),dimension=float(1/p),
                a=float(1-b),b=float(b),delta=float(delta_ratio*p),delta_ratio=delta_ratio,
                b_bound=b_bound,b_at_boundary=bool(abs(b)>.999*b_bound),
                p_at_boundary=bool(p<.0301 or p>.7999),
                objective=float(fit.fun@fit.fun),dof=len(target)-2,
                weighting="full_shared_reference_GLS" if covariance is not None else "unweighted",
                success=bool(fit.success),prediction=np.r_[prediction(fit.x),1.].tolist(),
                positive_predictions=bool(np.all(prediction(fit.x)>0)),
                normalization="k(n_ref)=1; a+b=1 imposed, unlike unconstrained free-a variant")
