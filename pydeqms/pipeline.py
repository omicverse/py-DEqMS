"""High-level pipeline mirroring DEqMS's reference workflow:

    fit = lmFit(M, design)
    fit = contrasts.fit(fit, makeContrasts(...))   # optional
    fit = eBayes(fit)
    fit$count = count
    fit = spectraCounteBayes(fit, fit.method='loess', coef_col=...)
    outputResult(fit, coef_col=...)

is collapsed into one ``deqms(M, design, count, contrast=...)`` call.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .deqms import output_result, spectra_count_ebayes
from .ebayes import ebayes
from .lmfit import LinearModelFit, lm_fit


ContrastLike = Union[np.ndarray, "list[float]", int, str, pd.DataFrame, None]


def _resolve_contrast(
    fit: LinearModelFit,
    contrast: ContrastLike,
) -> tuple[LinearModelFit, int]:
    """Apply contrasts.fit if a contrast vector is given, return (fit, coef_col).

    If ``contrast`` is an int or a string, it just picks that coefficient
    of the model (no contrasts.fit step).
    """
    if contrast is None:
        return fit, 0
    if isinstance(contrast, int):
        return fit, contrast
    if isinstance(contrast, str):
        if contrast not in fit.coef_names:
            raise KeyError(f"coef {contrast!r} not in design columns {fit.coef_names}")
        return fit, fit.coef_names.index(contrast)

    # Vector contrast (or matrix). Compute ``coef · C`` and rotate ``stdev.unscaled``.
    C = np.asarray(contrast, dtype=float)
    if C.ndim == 1:
        C = C[:, None]  # (n_coefs, 1)
    if C.shape[0] != fit.coefficients.shape[1]:
        raise ValueError(
            f"contrast has {C.shape[0]} entries but design has "
            f"{fit.coefficients.shape[1]} coefficients."
        )

    new_coefs = fit.coefficients @ C  # (n_proteins, n_contrasts)
    # stdev.unscaled rotates as sqrt(diag(C' (X'X)^{-1} C)).
    XtX = fit.design.T @ fit.design
    XtX_inv = np.linalg.pinv(XtX)
    cov_C = C.T @ XtX_inv @ C
    new_su_row = np.sqrt(np.maximum(np.diag(cov_C), 0.0))
    new_su = np.broadcast_to(new_su_row, new_coefs.shape).copy()

    fit.coefficients = new_coefs
    fit.stdev_unscaled = new_su
    fit.coef_names = [f"contrast_{i}" for i in range(C.shape[1])]
    return fit, 0


def deqms(
    M: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame,
    count: np.ndarray | pd.Series,
    *,
    contrast: ContrastLike = None,
    fit_method: str = "loess",
    loess_span: float = 0.75,
    run_ebayes: bool = True,
) -> pd.DataFrame:
    r"""End-to-end DEqMS workflow: ``lmFit → eBayes → spectraCounteBayes → outputResult``.

    Parameters
    ----------
    M
        Protein × sample log-intensity matrix. NaN = missing.
    design
        Sample × coefficient design matrix (one column per group, or an
        intercept + treatment column for a 2-group test).
    count
        Per-protein peptide / PSM count (used by the prior regression).
    contrast
        How to pick / form the test coefficient. Options:

        * ``None`` (default) — uses coefficient column 0.
        * ``int`` — coefficient column index.
        * ``str`` — coefficient column name (from ``design.columns``).
        * ``np.ndarray`` of shape ``(n_coefs,)`` or ``(n_coefs, k)`` —
          ``contrasts.fit``-style linear combination of the design columns.
    fit_method
        ``'loess'`` (default), ``'nls'``, or ``'spline'`` — same as R.
    loess_span
        ``span`` for the ``'loess'`` family.
    run_ebayes
        If ``True`` (default), also run the un-modified limma eBayes so
        the ``t`` / ``P.Value`` / ``adj.P.Val`` columns are populated
        side-by-side with the DEqMS-adjusted ``sca.*`` columns.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``sca.P.Value`` ascending; same columns as
        ``DEqMS::outputResult`` plus optional ``gene`` index.
    """
    fit = lm_fit(M, design)
    fit, coef_col = _resolve_contrast(fit, contrast)
    if run_ebayes:
        ebayes(fit)
    spectra_count_ebayes(
        fit, count=count, fit_method=fit_method, loess_span=loess_span,
    )
    return output_result(fit, coef_col=coef_col)
