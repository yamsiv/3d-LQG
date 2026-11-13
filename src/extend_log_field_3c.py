"""Sharp Fourier extensions retaining the original low-mode randomness.

The pure merge is testable without sampling. In production the proposal's
high modes must be independent of the retained field. Heat filters are not
supported because their low-mode variances change with resolution.
"""
from __future__ import annotations
import numpy as np
from .log_field_3d import TorusLogField,sample_field


def merge_retained_sharp_modes(retained,independent_proposal):
    if retained.metadata.get('regularization')!='sharp'or independent_proposal.metadata.get('regularization')!='sharp':
        raise ValueError('Exact low-mode retention requires sharp regularization')
    old_n=retained.h.shape[0];new_n=independent_proposal.h.shape[0]
    if new_n<=old_n:raise ValueError('Extension requires a larger grid')
    k=np.rint(np.fft.fftfreq(old_n)*old_n).astype(int)
    active=retained.spectrum>0
    i,j,ell=np.nonzero(active)
    destination=(k[i]%new_n,k[j]%new_n,k[ell]%new_n)
    proposed_spectrum=independent_proposal.spectrum[destination]
    if not np.array_equal(proposed_spectrum,retained.spectrum[active]):
        raise ValueError('Retained andnew low-mode variances must agree exactly')
    coefficients=np.fft.fftn(independent_proposal.h)/new_n**3
    retained_coefficients=np.fft.fftn(retained.h)/old_n**3
    coefficients[destination]=retained_coefficients[active]
    complex_values=np.fft.ifftn(coefficients*new_n**3)
    metadata=dict(independent_proposal.metadata,
        coupling='retained active sharp Fourier coefficients;independent proposal high modes',
        retained_sampling_n=old_n,retained_active_modes=int(active.sum()),
        independent_proposal_required=True,imaginary_roundoff_max=float(abs(complex_values.imag).max()))
    return TorusLogField(complex_values.real.copy(),independent_proposal.variance,
        independent_proposal.covariance,independent_proposal.spectrum,metadata)


def extend_sharp_field(retained,new_n,rng):
    """Future authorized simulation;use a separately seeded high-mode Generator."""
    proposal=sample_field(new_n,rng,regularization='sharp',
        cutoff_fraction=retained.metadata.get('cutoff_fraction',1/3))
    return merge_retained_sharp_modes(retained,proposal)
