"""Plotting helpers — Python ports of DEqMS plot functions.

* :func:`peptide_profile_plot` — per-protein peptide-intensity profile lines
  (port of ``DEqMS::peptideProfilePlot``).
* :func:`residual_plot` — residual-vs-fitted scatter from the count-prior
  fit inside a :class:`pydeqms.LinearModelFit` (port of ``DEqMS::Residualplot``).
* :func:`variance_boxplot` — boxplot of log(sigma²) by peptide count, with the
  count-prior fitted curve overlaid (port of ``DEqMS::VarianceBoxplot``).
* :func:`variance_scatter_plot` — scatter of log(sigma²) vs log2(count)
  with prior-fit overlay (port of ``DEqMS::VarianceScatterplot``).

All four lazy-import ``matplotlib`` so that ``import pydeqms`` is free
of a matplotlib dependency. Each helper accepts an optional ``ax``; if
``None`` a new ``Figure`` / ``Axes`` is created.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .lmfit import LinearModelFit


def _ensure_ax(ax):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    return ax


def peptide_profile_plot(
    peptide_df: pd.DataFrame,
    protein: str,
    protein_col: str,
    sample_cols: Sequence[str],
    peptide_col: Optional[str] = None,
    ax=None,
):
    r"""Plot per-PSM intensity profiles for one protein.

    Mirrors ``DEqMS::peptideProfilePlot``. One line per PSM, x = sample
    channel, y = intensity, color = peptide identifier.

    Parameters
    ----------
    peptide_df
        Peptide / PSM table.
    protein
        Protein identifier to plot (filters rows where
        ``peptide_df[protein_col] == protein``).
    protein_col
        Column name with the protein identifier.
    sample_cols
        Sample channel columns to draw on the x-axis (in order).
    peptide_col
        Column name with the peptide identifier (used for line colors /
        legend). If ``None``, the DataFrame index is used.
    ax
        Matplotlib axes to draw on; a new figure is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    sub = peptide_df.loc[peptide_df[protein_col] == protein].copy()
    if sub.empty:
        raise ValueError(f"no PSMs for protein {protein!r}")
    if peptide_col is None:
        sub["__peptide__"] = sub.index.astype(str)
        peptide_col = "__peptide__"

    ax = _ensure_ax(ax)
    samples = list(sample_cols)
    xs = np.arange(len(samples))
    cmap = plt.get_cmap("tab20")
    peptides = list(sub[peptide_col].astype(str).unique())
    color_map = {p: cmap(i % cmap.N) for i, p in enumerate(peptides)}
    for _, row in sub.iterrows():
        ys = row[samples].to_numpy(dtype=float)
        c = color_map[str(row[peptide_col])]
        ax.plot(xs, ys, marker="o", color=c, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(samples, rotation=90, ha="right")
    ax.set_xlabel("samples")
    ax.set_ylabel("log2 intensity")
    ax.set_title(str(protein))
    # Build a legend mapping peptide → color.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=color_map[p], marker="o", label=p)
        for p in peptides
    ]
    if len(handles) <= 20:
        ax.legend(handles=handles, title="Peptide", fontsize=7,
                  loc="best", frameon=False)
    return ax


def _residuals_from_fit(fit: LinearModelFit) -> tuple[np.ndarray, np.ndarray]:
    r"""Return (log2_count, residual_y) the same way R ``Residualplot`` does.

    R: nls model → ``log(fitted) - log(sigma^2)``;
       loess/spline → ``residuals(fit$model)`` which equals
       ``y_obs - y_pred = log(sigma²) - predicted_log(sigma²)``.

    We reconstruct this from :class:`LinearModelFit` fields:

    * ``loess`` — ``log_var - log(sca_priorvar)``.
    * ``spline`` — same as loess (R returns ``log(sigma²) - predicted``).
    * ``nls`` — ``log(predicted_sigma²) - log(sigma²)`` (R's sign convention).
    """
    if fit.count is None or fit.sca_priorvar is None:
        raise ValueError("fit has no count / sca_priorvar — run spectra_count_ebayes first.")
    sigma2 = np.asarray(fit.sigma, dtype=float) ** 2
    log_var = np.log(sigma2)
    s02 = np.asarray(fit.sca_priorvar, dtype=float)
    log_pred = np.log(s02)
    count = np.asarray(fit.count, dtype=float)
    x = np.log2(count)
    if fit.fit_method == "nls":
        y = log_pred - log_var
    else:
        y = log_var - log_pred
    return x, y


def residual_plot(fit: LinearModelFit, ax=None):
    r"""Scatter residuals from the count-prior fit (``DEqMS::Residualplot``).

    Parameters
    ----------
    fit
        :class:`LinearModelFit` with :func:`pydeqms.spectra_count_ebayes`
        already run (so ``fit.count`` / ``fit.sca_priorvar`` / ``fit.fit_method``
        are populated).
    ax
        Optional Axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    x, y = _residuals_from_fit(fit)
    ax = _ensure_ax(ax)
    good = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[good], y[good], s=5, c="k", alpha=0.6)
    ax.set_xlabel("log2(count)")
    ax.set_ylabel("Variance(fitted - observed)")
    return ax


def variance_boxplot(fit: LinearModelFit, n: int = 20, ax=None):
    r"""Boxplot of ``log(sigma²)`` by peptide count, with prior fit overlay.

    Mirrors ``DEqMS::VarianceBoxplot``. One box per unique peptide-count
    value (1..n); the red line is the count-prior model evaluated on
    ``seq(1, n)``.

    Parameters
    ----------
    fit
        :class:`LinearModelFit` after :func:`pydeqms.spectra_count_ebayes`.
    n
        Maximum peptide count to include (filters to ``count <= n``).
    ax
        Optional Axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if fit.count is None or fit.sca_priorvar is None:
        raise ValueError("fit has no count / sca_priorvar — run spectra_count_ebayes first.")
    count = np.asarray(fit.count, dtype=float)
    sigma2 = np.asarray(fit.sigma, dtype=float) ** 2

    mask = (count <= n) & np.isfinite(count) & np.isfinite(sigma2) & (sigma2 > 0)
    c = count[mask].astype(int)
    log_var = np.log(sigma2[mask])

    ax = _ensure_ax(ax)
    grid = np.arange(1, n + 1)
    data = [log_var[c == i] for i in grid]
    positions = grid
    # Some boxes may be empty — matplotlib needs non-empty lists.
    keep = [(p, d) for p, d in zip(positions, data) if d.size > 0]
    if keep:
        pos, dat = zip(*keep)
        ax.boxplot(list(dat), positions=list(pos), widths=0.6, showfliers=False)
    ax.set_xticks(grid)
    ax.set_xticklabels([str(i) for i in grid])
    ax.set_xlabel("count")
    ax.set_ylabel("log(Variance)")

    y_pred = _predict_prior(fit, grid)
    ax.plot(grid, y_pred, color="red", lw=2.5)
    return ax


def variance_scatter_plot(fit: LinearModelFit, n: int = 20, ax=None):
    r"""Scatter of ``log(sigma²)`` vs ``log2(count)`` with prior-fit overlay.

    Mirrors ``DEqMS::VarianceScatterplot``.

    Parameters
    ----------
    fit
        :class:`LinearModelFit` after :func:`pydeqms.spectra_count_ebayes`.
    n
        Unused for the scatter itself but accepted for signature parity
        with :func:`variance_boxplot`.
    ax
        Optional Axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if fit.count is None or fit.sca_priorvar is None:
        raise ValueError("fit has no count / sca_priorvar — run spectra_count_ebayes first.")
    count = np.asarray(fit.count, dtype=float)
    sigma2 = np.asarray(fit.sigma, dtype=float) ** 2
    good = np.isfinite(count) & (count > 0) & np.isfinite(sigma2) & (sigma2 > 0)
    x = np.log2(count[good])
    y = np.log(sigma2[good])

    ax = _ensure_ax(ax)
    ax.scatter(x, y, s=6, c="k", alpha=0.6)

    # Red overlay = prior-fit (sca.priorvar) sorted by count.
    s02 = np.asarray(fit.sca_priorvar, dtype=float)[good]
    order = np.argsort(count[good])
    ax.plot(np.log2(count[good][order]), np.log(s02[order]),
            color="red", lw=2.5)
    ax.set_xlabel("log2(count)")
    ax.set_ylabel("log(Variance)")
    return ax


def _predict_prior(fit: LinearModelFit, x_count: np.ndarray) -> np.ndarray:
    r"""Evaluate the count-prior model on user-provided peptide counts.

    For ``fit_method='loess'`` the stored ``skmisc.loess`` model is
    evaluated on ``log2(x_count)``; for ``'spline'`` the same. For
    ``'nls'`` we apply ``a + b/x`` and take ``log``.
    """
    x_count = np.asarray(x_count, dtype=float)
    method = fit.fit_method
    model = fit.loess_model
    if model is None:
        raise ValueError("fit has no loess_model; rerun spectra_count_ebayes.")
    if method == "loess":
        # skmisc loess rejects extrapolation by default — clip the
        # evaluation grid to the training range so end-cap counts work.
        x_log2 = np.log2(x_count)
        x_train = np.asarray(model.inputs.x, dtype=float).ravel()
        lo, hi = float(x_train.min()), float(x_train.max())
        x_clip = np.clip(x_log2, lo, hi)
        pred = model.predict(x_clip, stderror=False)
        return np.asarray(pred.values, dtype=float)
    if method == "spline":
        return np.asarray(model(np.log2(x_count)), dtype=float)
    if method == "nls":
        a, b = float(model["a"]), float(model["b"])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(a + b / x_count)
    raise ValueError(f"unknown fit_method {method!r}")
