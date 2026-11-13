"""Mean-zero, unit-log torus Gaussian fields with shared Fourier randomness.

The covariance of the unregularized distribution is
2*pi**2 * (-Delta)**(-3/2), on the unit three-torus. Integer Fourier modes
therefore have variance S(k)=1/(4*pi*|k|**3). Finite grids contain a
regularized continuum spectrum, NOT the inverse lattice Laplacian.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class TorusLogField:
    h: np.ndarray
    variance: float
    covariance: np.ndarray
    spectrum: np.ndarray
    metadata: dict


@lru_cache(maxsize=24)
def _spectral_configuration(n, regularization, cutoff_fraction):
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("Grid size must be an integer >=4")
    if not 0 < cutoff_fraction < .5:
        raise ValueError("Require 0<cutoff_fraction<1/2 to exclude Nyquist modes")
    k = np.rint(np.fft.fftfreq(n)*n).astype(int)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    radius = np.sqrt(kx*kx+ky*ky+kz*kz)
    nonzero = radius > 0
    spectrum = np.zeros_like(radius)
    spectrum[nonzero] = 1/(4*np.pi*radius[nonzero]**3)
    cutoff = n*cutoff_fraction
    if regularization == "sharp":
        spectrum[radius >= cutoff] = 0.
    elif regularization == "heat":
        # Square of field multiplier exp(-|k|^2/(2*cutoff^2)). The
        # strict cube cap keeps the field real for even grids; cap sensitivity
        # is a finite-resolution systematic, explicitly retained in metadata.
        spectrum *= np.exp(-(radius/cutoff)**2)
        spectrum[(abs(kx) >= n/2) | (abs(ky) >= n/2) | (abs(kz) >= n/2)] = 0.
    else:
        raise ValueError("regularization must be 'sharp' or 'heat'")
    covariance = np.fft.ifftn(spectrum*n**3).real
    variance = float(spectrum.sum())
    spectrum.setflags(write=False)
    covariance.setflags(write=False)
    k.setflags(write=False)
    return k, spectrum, covariance, variance


def spectral_configuration(n, *, regularization="sharp", cutoff_fraction=1/3):
    """Return immutable S(k), covariance C(j/N), and exact point variance."""
    _, spectrum, covariance, variance = _spectral_configuration(n, regularization, cutoff_fraction)
    return spectrum, covariance, variance


def sample_coupled_fields(sizes, rng=None, *, regularization="sharp", cutoff_fraction=1/3):
    """Return {N: TorusLogField}, sharing every available Gaussian mode.

    A real iid N(0,1) array on the largest grid has unitary Fourier transform
    z_k with E|z_k|^2=1 and z_-k=conj(z_k). Distinct nonconjugate complex
    modes are independent, with component variance 1/2. Smaller grids select
    those same integer modes, before their scale-dependent regularization.
    At each N, h=ifft(N^3*sqrt(S_N)*z). The zero mode is exactly removed.

    Coupling is reproducible for a fixed requested tuple and seed. Asking
    for a different maximum N changes the FFT representation of the common
    Gaussian randomness; seeds alone are not a cross-call nesting contract.
    """
    sizes = tuple(sorted(set(sizes)))
    if not sizes:
        raise ValueError("At least one grid size required")
    configs = {n: _spectral_configuration(n, regularization, cutoff_fraction) for n in sizes}
    rng = np.random.default_rng() if rng is None else rng
    largest = max(sizes)
    white = rng.normal(size=(largest,)*3)
    noise = np.fft.fftn(white)/np.sqrt(largest**3)
    fields = {}
    for n, (k, spectrum, covariance, variance) in configs.items():
        selected = noise[np.ix_(k % largest, k % largest, k % largest)]
        coefficients = np.sqrt(spectrum)*selected
        complex_values = np.fft.ifftn(coefficients*n**3)
        imaginary = float(abs(complex_values.imag).max())
        h = complex_values.real.copy()
        metadata = {
            "model": "unit_log_continuum_torus_fourier_cutoff", "dimension": 3,
            "n": int(n), "domain": "unit three-torus [0,1)^3",
            "regularization": regularization, "cutoff_fraction": float(cutoff_fraction),
            "cutoff_integer_radius": float(n*cutoff_fraction),
            "point_variance": variance, "zero_mode": "removed",
            "spectrum_constant": float(1/(4*np.pi)),
            "covariance_operator_constant": float(2*np.pi**2),
            "paper_a3": float(1/(2*np.pi**2)),
            "fft": "numpy unnormalized forward, 1/N^3 inverse",
            "shared_resolutions": list(sizes), "gaussian_source_grid": int(largest),
            "imaginary_roundoff_max": imaginary,
            "not_discrete_laplacian": True,
        }
        fields[n] = TorusLogField(h, variance, covariance, spectrum, metadata)
    return fields


def sample_field(n, rng=None, **kwargs):
    """Single-resolution convenience wrapper; see coupling contract above."""
    return sample_coupled_fields([n], rng, **kwargs)[n]


def resample_field(field, n):
    """Evaluate the identical band-limited field on a finer quadrature grid.

    No modes, randomness or point variance are added. This separates GMC or
    metric discretization from increasing the underlying field cutoff.
    """
    old_n = field.h.shape[0]
    if not isinstance(n, (int, np.integer)) or n < old_n:
        raise ValueError("Resampling requires an integer n>=original grid size")
    k = np.rint(np.fft.fftfreq(old_n)*old_n).astype(int)
    # Modes at an even-grid Nyquist frequency were excluded originally.
    active = field.spectrum > 0
    ii, jj, kk = np.nonzero(active)
    destination = (k[ii] % n, k[jj] % n, k[kk] % n)
    old_coefficients = np.fft.fftn(field.h)/old_n**3
    coefficients = np.zeros((n,)*3, complex)
    coefficients[destination] = old_coefficients[ii, jj, kk]
    spectrum = np.zeros((n,)*3)
    spectrum[destination] = field.spectrum[ii, jj, kk]
    h = np.fft.ifftn(coefficients*n**3).real
    covariance = np.fft.ifftn(spectrum*n**3).real
    metadata = dict(field.metadata, n=int(n), same_field_quadrature=True,
                    original_sampling_n=old_n,
                    cutoff_integer_radius=field.metadata["cutoff_integer_radius"])
    return TorusLogField(h, field.variance, covariance, spectrum, metadata)


def log_coefficient_fit(n=128, *, regularization="sharp", cutoff_fraction=1/3,
                        distance_window=(.025, .125)):
    """Deterministic axial covariance slope over a declared finite window.

    This is a finite-window diagnostic, not a consistent slope estimator
    at fixed N or a numerical proof of the continuum covariance asymptotic.
    """
    _, covariance, _ = spectral_configuration(n, regularization=regularization,
                                              cutoff_fraction=cutoff_fraction)
    j = np.arange(1, n//2+1)
    r = j/n
    selected = (r >= distance_window[0]) & (r <= distance_window[1])
    if selected.sum() < 3:
        raise ValueError("Fit window contains fewer than three separations")
    slope, intercept = np.polyfit(-np.log(r[selected]), covariance[j[selected], 0, 0], 1)
    return {"n": int(n), "window": list(distance_window), "slope": float(slope),
            "intercept": float(intercept), "points": int(selected.sum())}
