"""
Minimal port of ``limma::lmFit`` — fit a linear model independently
per protein (row), given a design matrix.

This is the upstream step DEqMS's ``spectraCounteBayes`` consumes. We
only implement the no-weight, no-correlation OLS path that the canonical
DEqMS workflow uses (`lmFit(M, design)` on a log-intensity matrix).

The returned object exposes the fields ``spectraCounteBayes`` needs:
``coefficients``, ``sigma``, ``df.residual``, ``stdev.unscaled``, plus
``Amean`` (per-protein mean) for downstream tabulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LinearModelFit:
    """Container mirroring the ``MArrayLM`` object returned by limma."""

    coefficients: np.ndarray                # (n_proteins, n_coefs)
    stdev_unscaled: np.ndarray              # (n_proteins, n_coefs)
    sigma: np.ndarray                       # (n_proteins,) residual SD
    df_residual: np.ndarray                 # (n_proteins,) per-protein DoF
    Amean: np.ndarray                       # (n_proteins,) row-mean
    design: np.ndarray                      # (n_samples, n_coefs)
    coef_names: list[str] = field(default_factory=list)
    protein_names: list[str] = field(default_factory=list)
    # DEqMS extras (populated by ``spectra_count_ebayes``)
    count: Optional[np.ndarray] = None      # (n_proteins,) peptide / PSM count
    t: Optional[np.ndarray] = None          # eBayes moderated t
    p_value: Optional[np.ndarray] = None
    s2_post: Optional[np.ndarray] = None
    df_prior: Optional[float] = None
    s2_prior: Optional[float] = None
    df_total: Optional[np.ndarray] = None
    sca_t: Optional[np.ndarray] = None
    sca_p: Optional[np.ndarray] = None
    sca_postvar: Optional[np.ndarray] = None
    sca_priorvar: Optional[np.ndarray] = None
    sca_dfprior: Optional[float] = None
    fit_method: Optional[str] = None
    loess_model: Optional[object] = None


def lm_fit(
    M: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame,
) -> LinearModelFit:
    r"""Fit y = X β + ε independently per protein.

    Parameters
    ----------
    M
        Log-intensity matrix, rows = proteins, columns = samples.
        ``NaN`` entries are treated as missing and dropped on a per-row
        basis (matching ``limma::lmFit`` which uses ``na.action='na.exclude'``
        when fitting).
    design
        Design matrix, rows = samples, columns = coefficients (groups).

    Returns
    -------
    LinearModelFit
        ``coefficients``, ``sigma``, ``df_residual``, ``stdev_unscaled``,
        ``Amean`` populated; DEqMS / eBayes fields left as ``None``.

    Notes
    -----
    For a perfect ``limma::lmFit`` match (which uses a QR-pivoting
    rank-revealing decomposition for the design), set ``design`` to a
    full-rank matrix. We do the same QR up front and reuse it across
    rows when no missing values per row exist (the no-NA fast path is
    O(n_proteins × n_samples × n_coefs) — vectorised in NumPy).
    """
    if isinstance(M, pd.DataFrame):
        protein_names = list(M.index.astype(str))
        sample_names = list(M.columns.astype(str))
        M_arr = M.to_numpy(dtype=float)
    else:
        M_arr = np.asarray(M, dtype=float)
        protein_names = [f"prot_{i}" for i in range(M_arr.shape[0])]
        sample_names = [f"S{j}" for j in range(M_arr.shape[1])]

    if isinstance(design, pd.DataFrame):
        coef_names = list(design.columns.astype(str))
        X = design.to_numpy(dtype=float)
    else:
        X = np.asarray(design, dtype=float)
        coef_names = [f"X{j}" for j in range(X.shape[1])]

    if X.shape[0] != M_arr.shape[1]:
        raise ValueError(
            f"design has {X.shape[0]} rows but M has {M_arr.shape[1]} columns; "
            "design rows must match M columns (samples)."
        )

    n_proteins, n_samples = M_arr.shape
    n_coefs = X.shape[1]

    has_missing = np.isnan(M_arr).any(axis=1)

    # ---- Pre-compute the no-missing path (vectorised). -------------- #
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    Hat = X @ XtX_inv @ X.T  # n_samples × n_samples
    coefs_all = (XtX_inv @ X.T @ M_arr.T).T  # (n_proteins, n_coefs)
    fitted_all = (X @ coefs_all.T).T          # (n_proteins, n_samples)
    residuals_all = M_arr - fitted_all
    df_full = n_samples - np.linalg.matrix_rank(X)
    sse_all = np.nansum(residuals_all ** 2, axis=1)
    sigma_all = np.sqrt(sse_all / df_full) if df_full > 0 else np.full(n_proteins, np.nan)

    # stdev.unscaled is sqrt of diagonal of (X'X)^-1 — same across proteins
    # when no missing data is present.
    stdev_unscaled_row = np.sqrt(np.maximum(np.diag(XtX_inv), 0.0))

    coefficients = coefs_all
    sigma = sigma_all
    df_residual = np.full(n_proteins, float(df_full))
    stdev_unscaled = np.broadcast_to(stdev_unscaled_row, (n_proteins, n_coefs)).copy()

    # ---- Per-row recompute for proteins with NaN. ------------------- #
    if has_missing.any():
        for i in np.flatnonzero(has_missing):
            y = M_arr[i]
            keep = ~np.isnan(y)
            X_i = X[keep]
            y_i = y[keep]
            n_i = y_i.size
            if n_i == 0:
                coefficients[i] = np.nan
                sigma[i] = np.nan
                df_residual[i] = 0.0
                stdev_unscaled[i] = np.nan
                continue
            try:
                XtX_i = X_i.T @ X_i
                XtX_inv_i = np.linalg.pinv(XtX_i)
                rank_i = np.linalg.matrix_rank(X_i)
                beta_i = XtX_inv_i @ X_i.T @ y_i
                resid_i = y_i - X_i @ beta_i
                df_i = n_i - rank_i
                sse_i = float(np.sum(resid_i ** 2))
                sig_i = np.sqrt(sse_i / df_i) if df_i > 0 else np.nan
                coefficients[i] = beta_i
                sigma[i] = sig_i
                df_residual[i] = float(df_i)
                stdev_unscaled[i] = np.sqrt(np.maximum(np.diag(XtX_inv_i), 0.0))
            except np.linalg.LinAlgError:
                coefficients[i] = np.nan
                sigma[i] = np.nan
                df_residual[i] = 0.0
                stdev_unscaled[i] = np.nan

    Amean = np.nanmean(M_arr, axis=1)

    return LinearModelFit(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=df_residual,
        Amean=Amean,
        design=X,
        coef_names=coef_names,
        protein_names=protein_names,
    )
