"""Smoke tests for the plotting helpers — Axes returned, no crash."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pydeqms import (
    lm_fit,
    peptide_profile_plot,
    residual_plot,
    spectra_count_ebayes,
    variance_boxplot,
    variance_scatter_plot,
)


def _make_fit(seed: int = 0, n_proteins: int = 200, n_per_group: int = 6,
              fit_method: str = "loess"):
    rng = np.random.default_rng(seed)
    n_samples = n_per_group * 2
    base = rng.normal(20.0, 1.5, size=n_proteins)
    count = rng.integers(1, 20, size=n_proteins).astype(float)
    sigma = 0.4 + 1.0 / np.sqrt(count)
    M = np.zeros((n_proteins, n_samples))
    for g in range(2):
        M[:, g * n_per_group:(g + 1) * n_per_group] = rng.normal(
            loc=base[:, None], scale=sigma[:, None],
            size=(n_proteins, n_per_group),
        )
    M_df = pd.DataFrame(M,
                        index=[f"prot_{i:04d}" for i in range(n_proteins)],
                        columns=[f"S{j:02d}" for j in range(n_samples)])
    design = np.zeros((n_samples, 2))
    design[:n_per_group, 0] = 1.0
    design[n_per_group:, 1] = 1.0
    fit = lm_fit(M_df, design)
    spectra_count_ebayes(fit, count=count, fit_method=fit_method)
    return fit, count


def test_residual_plot_returns_axes():
    fit, _ = _make_fit()
    ax = residual_plot(fit)
    assert ax is not None
    assert hasattr(ax, "plot")


def test_residual_plot_nls():
    fit, _ = _make_fit(fit_method="nls")
    ax = residual_plot(fit)
    assert ax is not None


def test_variance_boxplot_returns_axes():
    fit, _ = _make_fit()
    ax = variance_boxplot(fit, n=20)
    assert ax is not None


def test_variance_boxplot_nls():
    fit, _ = _make_fit(fit_method="nls")
    ax = variance_boxplot(fit, n=15)
    assert ax is not None


def test_variance_scatter_plot_returns_axes():
    fit, _ = _make_fit()
    ax = variance_scatter_plot(fit)
    assert ax is not None


def test_variance_scatter_plot_spline():
    fit, _ = _make_fit(fit_method="spline")
    ax = variance_scatter_plot(fit)
    assert ax is not None


def test_peptide_profile_plot_returns_axes():
    rng = np.random.default_rng(0)
    n_psms = 60
    proteins = rng.choice(["A", "B", "C"], size=n_psms)
    peptides = [f"pep_{i}" for i in range(n_psms)]
    sample_cols = [f"S{j+1:02d}" for j in range(8)]
    intensities = rng.normal(20.0, 1.0, size=(n_psms, 8))
    df = pd.DataFrame(intensities, columns=sample_cols)
    df["peptide"] = peptides
    df["protein"] = proteins
    ax = peptide_profile_plot(df, protein="A", protein_col="protein",
                              sample_cols=sample_cols, peptide_col="peptide")
    assert ax is not None
    assert ax.get_title() == "A"


def test_peptide_profile_plot_raises_for_missing_protein():
    df = pd.DataFrame({
        "peptide": ["p1"], "protein": ["A"], "S1": [1.0], "S2": [2.0],
    })
    with pytest.raises(ValueError, match="no PSMs"):
        peptide_profile_plot(df, "Z", "protein", ["S1", "S2"],
                             peptide_col="peptide")
