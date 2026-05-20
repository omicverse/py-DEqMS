"""
Port of ``limma::eBayes`` — Smyth 2004 empirical-Bayes moderated t.

Estimates a single ``(d0, s0²)`` prior across all proteins by matching
trigamma moments, then plugs the posterior variance into the t-test for
each protein. The downstream ``DEqMS::spectraCounteBayes`` replaces the
flat prior by a count-conditional prior; see ``deqms.py``.

References
----------
Smyth GK. *Linear Models and Empirical Bayes Methods for Assessing
Differential Expression in Microarray Experiments.* Statistical
Applications in Genetics and Molecular Biology, 2004, 3 (1): 3.
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats

from .lmfit import LinearModelFit


def _trigamma_inverse(x: float, *, max_iter: int = 50,
                      atol: float = 1e-8) -> float:
    """Newton solve for ``y`` such that ``trigamma(y) = x``.

    Equivalent to ``limma::trigammaInverse``.
    """
    if x <= 0:
        return float("inf")
    # Initial estimate: trigamma(y) ≈ 1/y + 1/(2y²), so y ≈ 0.5 + 1/x for
    # small x; trigamma(y) ≈ 1/y for large y. limma uses these branches.
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    # Newton update matching ``limma::trigammaInverse``:
    #   dif = trigamma(y) * (1 - trigamma(y) / x) / psigamma(y, deriv=2)
    #   y   = y + dif
    for _ in range(max_iter):
        tg = special.polygamma(1, y)
        d = tg * (1 - tg / x) / special.polygamma(2, y)
        y += d
        if abs(d / y) < atol:
            break
    return float(y)


def ebayes(
    fit: LinearModelFit,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
) -> LinearModelFit:
    r"""Empirical-Bayes moderated t-statistics (Smyth 2004).

    Mutates ``fit`` in place — same convention as ``limma::eBayes(fit)``
    — and returns the augmented object.

    The augmented fields are:

    * ``df_prior`` (scalar) — prior degrees of freedom ``d0``.
    * ``s2_prior`` (scalar) — prior variance ``s0²``.
    * ``s2_post`` (vector) — posterior variance per protein.
    * ``df_total`` (vector) — posterior degrees of freedom ``d0 + d_g``.
    * ``t``, ``p_value`` (matrices) — moderated t and two-sided p.

    Parameters
    ----------
    fit
        Output of :func:`pydeqms.lm_fit`.
    proportion
        Assumed proportion of differentially expressed proteins.
        Unused for ``t`` / ``p_value`` (matters only for ``B`` log-odds),
        but kept for API parity.
    stdev_coef_lim
        Clipping range for the per-coef ``stdev.unscaled``. limma uses
        ``c(0.1, 4)`` to guard against degenerate design columns.
    """
    sigma = np.asarray(fit.sigma, dtype=float)
    df = np.asarray(fit.df_residual, dtype=float)
    s2 = sigma ** 2

    # Valid proteins: positive df and finite, positive s2.
    good = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    if not good.any():
        raise ValueError("No proteins with positive residual variance — cannot run eBayes.")

    z = np.log(s2[good]) - special.digamma(df[good] / 2.0) + np.log(df[good] / 2.0)
    z_mean = float(np.mean(z))
    z_var = float(np.var(z, ddof=1))
    tg_df = special.polygamma(1, df[good] / 2.0)
    e_trigamma = float(np.mean(tg_df))
    # Smyth eq. (6): var(z) ≈ trigamma(d_g/2) + trigamma(d0/2)
    target = z_var - e_trigamma
    if target <= 0:
        d0 = float("inf")
    else:
        d0 = 2.0 * _trigamma_inverse(target)
    if not np.isfinite(d0):
        # Infinite prior df ⇒ posterior variance ≡ s0² ≡ mean of s2.
        s2_prior = float(np.mean(s2[good]))
    else:
        s2_prior = float(np.exp(z_mean + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))

    if np.isfinite(d0):
        s2_post = (d0 * s2_prior + df * s2) / (d0 + df)
        df_total = d0 + df
    else:
        s2_post = np.full_like(s2, s2_prior)
        df_total = np.full_like(df, np.inf)

    # Moderated t. stdev_unscaled may be NaN for completely-missing rows.
    su = np.asarray(fit.stdev_unscaled, dtype=float)
    su = np.clip(su, stdev_coef_lim[0], stdev_coef_lim[1])
    se = su * np.sqrt(s2_post)[:, None]
    coef = np.asarray(fit.coefficients, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = coef / se
    # Two-sided p — survival function of |t| on df_total dof.
    df_t = np.tile(df_total[:, None], (1, coef.shape[1]))
    with np.errstate(invalid='ignore'):
        p_value = 2.0 * stats.t.sf(np.abs(t), df_t)

    fit.df_prior = float(d0)
    fit.s2_prior = float(s2_prior)
    fit.s2_post = s2_post
    fit.df_total = df_total
    fit.t = t
    fit.p_value = p_value
    return fit
