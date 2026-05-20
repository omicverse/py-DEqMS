"""
Port of DEqMS's ``spectraCounteBayes`` (Zhu 2020, MCP).

The standard limma ``eBayes`` collapses every protein's residual
variance to a single ``(d0, s0²)`` prior. DEqMS replaces that flat
prior by a count-conditional prior: ``log(s²)`` is regressed on
``log2(peptide count)`` (loess by default) and the posterior variance
borrows strength from proteins with comparable peptide / PSM counts.

References
----------
Zhu Y., Orre L.M., Tran Y.Z., Mermelekas G., Johansson H.J.,
Malyutina A., Anders S., Lehtiö J.
*DEqMS: A method for accurate variance estimation in differential
protein expression analysis.*
Mol Cell Proteomics, 2020, 19 (6): 1047-1057.
DOI: 10.1074/mcp.TIR119.001646
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import special, stats

from .ebayes import _trigamma_inverse
from .lmfit import LinearModelFit


def spectra_count_ebayes(
    fit: LinearModelFit,
    count: np.ndarray | pd.Series,
    fit_method: str = "loess",
    loess_span: float = 0.75,
) -> LinearModelFit:
    r"""DEqMS posterior moderated t with peptide-count-aware prior.

    Mirrors ``DEqMS::spectraCounteBayes(fit, fit.method, coef_col)``;
    the ``coef_col`` argument is irrelevant for the variance fit (the
    prior is computed once across all coefs) so the Python API operates
    on the whole coefficient matrix.

    Parameters
    ----------
    fit
        A :class:`LinearModelFit` returned by :func:`pydeqms.lm_fit`;
        the standard ``eBayes`` step is NOT a prerequisite — DEqMS
        reads ``fit.sigma`` / ``fit.df_residual`` / ``fit.coefficients``
        / ``fit.stdev_unscaled`` directly.
    count
        Per-protein peptide or PSM count. Same length as
        ``fit.protein_names`` (rows of ``fit.coefficients``).
    fit_method
        Regression family. ``'loess'`` (R default, span 0.75) is the
        most-used; ``'nls'`` (``y = a + b/x``) and ``'spline'`` (smoothing
        spline) are accepted.
    loess_span
        ``span=`` for ``'loess'``. Default 0.75 (R DEqMS default).

    Returns
    -------
    LinearModelFit
        With ``sca_t``, ``sca_p``, ``sca_postvar``, ``sca_priorvar``,
        ``sca_dfprior``, ``fit_method`` and ``loess_model`` populated.
    """
    count = np.asarray(count, dtype=float)
    if count.size != fit.sigma.size:
        raise ValueError(
            f"count length {count.size} != n_proteins {fit.sigma.size}"
        )
    sigma2 = fit.sigma ** 2
    df = np.asarray(fit.df_residual, dtype=float)
    df = df.copy()
    df[df == 0] = np.nan

    good = np.isfinite(sigma2) & (sigma2 > 0) & np.isfinite(df) & (df > 0)
    if good.sum() < 5:
        raise ValueError(
            "Too few proteins with positive variance to fit a count-prior."
        )

    log_var = np.log(sigma2[good])
    df_good = df[good]
    eg = log_var - special.digamma(df_good / 2.0) + np.log(df_good / 2.0)

    x_good = count[good]
    loess_model = None
    if fit_method == "loess":
        from skmisc.loess import loess as _loess  # type: ignore
        x_log2 = np.log2(x_good)
        model = _loess(
            x_log2, log_var, span=loess_span,
        )
        model.fit()
        y_pred_good = np.asarray(model.outputs.fitted_values, dtype=float)
        loess_model = model
    elif fit_method == "nls":
        # R: nls(y ~ a + b/x), with y = sigma², x = count.
        from scipy.optimize import curve_fit
        def _nls_model(x, a, b):
            return a + b / x
        popt, _ = curve_fit(
            _nls_model, x_good, sigma2[good], p0=(0.1, 0.05), maxfev=10_000,
        )
        y_pred_good = np.log(_nls_model(x_good, *popt))
        loess_model = {"a": float(popt[0]), "b": float(popt[1])}
    elif fit_method == "spline":
        from scipy.interpolate import UnivariateSpline
        x_log2 = np.log2(x_good)
        order = np.argsort(x_log2)
        sp = UnivariateSpline(x_log2[order], log_var[order])
        # Evaluate at original (unsorted) positions for ``y.pred``.
        y_pred_good = np.asarray(sp(x_log2), dtype=float)
        loess_model = sp
    else:
        raise ValueError(f"unknown fit_method={fit_method!r}; choose loess/nls/spline")

    egpred = y_pred_good - special.digamma(df_good / 2.0) + np.log(df_good / 2.0)
    myfct = (eg - egpred) ** 2 - special.polygamma(1, df_good / 2.0)
    mean_myfct = float(np.nanmean(myfct))

    # R uses a grid search ``testd0 = i/10`` up to ``10 * numgenes`` to
    # find the smallest ``trigamma(d0/2)`` distance. We invert
    # ``trigamma`` analytically — same answer up to FP, no grid.
    if mean_myfct <= 0:
        d0 = float("inf")
    else:
        d0 = 2.0 * _trigamma_inverse(mean_myfct)

    if np.isfinite(d0):
        s02_good = np.exp(egpred + special.digamma(d0 / 2.0) - np.log(d0 / 2.0))
    else:
        s02_good = np.exp(egpred + 0 - np.log(1.0))  # graceful — won't reach

    # Map back to the full vector (NaN where invalid).
    s02_full = np.full_like(sigma2, np.nan, dtype=float)
    s02_full[good] = s02_good

    if np.isfinite(d0):
        post_var = (d0 * s02_full + df * sigma2) / (d0 + df)
        post_df = d0 + df
    else:
        post_var = s02_full
        post_df = np.full_like(df, np.inf)

    su = np.asarray(fit.stdev_unscaled, dtype=float)
    coef = np.asarray(fit.coefficients, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sca_t = coef / (su * np.sqrt(post_var)[:, None])
    df_t = np.tile(post_df[:, None], (1, coef.shape[1]))
    with np.errstate(invalid="ignore"):
        sca_p = 2.0 * stats.t.sf(np.abs(sca_t), df_t)

    fit.count = count
    fit.sca_t = sca_t
    fit.sca_p = sca_p
    fit.sca_postvar = post_var
    fit.sca_priorvar = s02_full
    fit.sca_dfprior = float(d0)
    fit.fit_method = fit_method
    fit.loess_model = loess_model
    return fit


def output_result(fit: LinearModelFit, coef_col: int = 0,
                  *, name: Optional[str] = None) -> pd.DataFrame:
    r"""Build the topTable-style result for one coefficient.

    Mirrors ``DEqMS::outputResult(fit, coef_col)``; the returned frame
    is sorted by ``sca.P.Value`` ascending.

    Columns
    -------
    ``gene``     — protein identifier (row name)
    ``logFC``    — coefficient at ``coef_col``
    ``AveExpr``  — per-protein mean (``fit.Amean``)
    ``t``        — limma moderated t (from :func:`pydeqms.ebayes`,
                   if it was run)
    ``P.Value``  — limma moderated p
    ``adj.P.Val``— BH-corrected p (limma path)
    ``count``    — peptide / PSM count
    ``sca.t``    — DEqMS moderated t
    ``sca.P.Value`` — DEqMS p
    ``sca.adj.pval`` — BH-corrected p (DEqMS path)
    """
    if fit.sca_t is None or fit.sca_p is None:
        raise ValueError("run spectra_count_ebayes(fit, count) first.")
    coef_name = fit.coef_names[coef_col] if fit.coef_names else f"coef_{coef_col}"
    sca_t = fit.sca_t[:, coef_col]
    sca_p = fit.sca_p[:, coef_col]
    sca_adj = stats.false_discovery_control(np.clip(sca_p, 0.0, 1.0), method="bh")
    logfc = fit.coefficients[:, coef_col]

    cols = {
        "gene": fit.protein_names,
        "logFC": logfc,
        "AveExpr": fit.Amean,
    }
    if fit.t is not None and fit.p_value is not None:
        cols["t"] = fit.t[:, coef_col]
        cols["P.Value"] = fit.p_value[:, coef_col]
        adj_pv = stats.false_discovery_control(
            np.clip(fit.p_value[:, coef_col], 0.0, 1.0), method="bh",
        )
        cols["adj.P.Val"] = adj_pv
    if fit.count is not None:
        cols["count"] = fit.count
    cols["sca.t"] = sca_t
    cols["sca.P.Value"] = sca_p
    cols["sca.adj.pval"] = sca_adj

    df = pd.DataFrame(cols)
    df = df.sort_values("sca.P.Value").reset_index(drop=True)
    if name is not None:
        df.attrs["coef_name"] = coef_name
    return df
