"""Finite field metric summaries shared by new3C confirmation producers."""
from __future__ import annotations
import numpy as np
from .gmc_3d import gmc_weights
from .lfpp_3d import build_lfpp_graph,root_distances
from .lfpp_periodic_3c import build_periodic_lfpp_graph
from .log_field_3d import spectral_configuration
from .metric_measure_3a import weighted_root_cdfs
from .phase3c_fields import sampled_points


def control_field(h,control,*,smooth_radius=2.):
    h=np.asarray(h,float);n=h.shape[0]
    spectrum,_,variance=spectral_configuration(n)
    if control=='log':return h,variance,{'transform':'identity','source_point_variance':variance}
    if control=='flat':return np.zeros_like(h),0.,{'transform':'zero deterministic field;xi=gamma=0','source_point_variance':variance}
    if control!='smooth':raise ValueError('Control must be log,smooth or flat')
    k=np.fft.fftfreq(n)*n
    radius2=sum(x*x for x in np.meshgrid(k,k,k,indexing='ij'))
    mask=(radius2>0)&(radius2<=smooth_radius**2)
    low_variance=float(spectrum[mask].sum())
    if low_variance<=0:raise ValueError('Smooth filter contains no nonzero mode')
    scale=float(np.sqrt(variance/low_variance))
    values=np.fft.ifftn(np.fft.fftn(h)*mask).real*scale
    return values,variance,{'transform':'low Fourier projection, rescaled to original deterministic point variance',
        'smooth_radius':smooth_radius,'unscaled_low_variance':low_variance,'multiplicative_scale':scale,
        'source_point_variance':variance,'not_log_correlated':True,
        'measure_type':'smooth exponential probability measure;not a log-field GMC limit'}


def metric_arrays(h,xi,variance,probes,uniforms,*,stencil='primitive2',bins=1024,
                  builder='periodic_kernel',cached=None):
    """Compute only gamma rows missing from a previously validated cache."""
    if builder not in['periodic_kernel','reference']:raise ValueError('Unknown builder')
    numerical={}if cached is None else dict(cached)
    measures={gamma:gmc_weights(h,gamma,variance)for _,gamma in probes}
    points=[sampled_points(measures[gamma].probabilities,uniforms)for _,gamma in probes]
    missing=[gamma for gamma in measures if gamma not in numerical]
    metadata={'stencil':stencil,'boundary':'torus','builder':builder,'new_unique_gamma_rows':len(missing)}
    if missing:
        roots=np.unique(np.concatenate([points[next(i for i,(_,g)in enumerate(probes)if g==gamma)]for gamma in missing]))
        function=build_periodic_lfpp_graph if builder=='periodic_kernel'else build_lfpp_graph
        graph,metadata=function(h,xi,stencil=stencil,boundary='torus')
        distances=root_distances(graph,roots)
        for gamma in missing:
            selected=points[next(i for i,(_,g)in enumerate(probes)if g==gamma)]
            d=distances[np.searchsorted(roots,selected)]
            numerical[gamma]={**weighted_root_cdfs(d,measures[gamma].probabilities.ravel(),bins=bins),
                'matrix':d[:,selected],'source_metric':None,'source_probe_index':None,'source_metric_sha256':None}
    masses=[]
    for(band,gamma),selected in zip(probes,points):
        measure=measures[gamma];data=numerical[gamma]
        masses.append({'band':band,'gamma':gamma,'total_mass':measure.total_mass,
            'log_total_mass':measure.log_total_mass,'collision_probability':float(np.sum(measure.probabilities**2)),
            'maximum_atom':float(measure.probabilities.max()),
            'observed_duplicate_point_fraction':float(1-len(np.unique(selected))/len(selected)),
            'source_metric':data.get('source_metric'),'source_probe_index':data.get('source_probe_index'),
            'source_metric_sha256':data.get('source_metric_sha256')})
    arrays={'grid':np.stack([numerical[g]['grid']for _,g in probes]),
        'cdfs':np.stack([numerical[g]['cdfs']for _,g in probes]).astype(np.float32),
        'root_means':np.stack([numerical[g]['root_means']for _,g in probes]),
        'matrices':np.stack([numerical[g]['matrix']for _,g in probes]),'points':np.stack(points),
        'bin_width':np.array([numerical[g]['distance_bin_width']for _,g in probes])}
    return arrays,masses,metadata,{'new_unique_gamma_rows':len(missing),
        'reused_unique_gamma_rows':len(measures)-len(missing)}
