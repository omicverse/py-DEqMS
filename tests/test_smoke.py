"""Basic sanity tests for the DEqMS pipeline — no R needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pydeqms import deqms, ebayes, lm_fit, spectra_count_ebayes


def _make_dataset(seed: int = 0, n_proteins: int = 500, n_per_group: int = 6,
                  n_groups: int = 2, prop_de: float = 0.1,
                  effect_size: float = 1.0):
    """Synthetic log2 intensity matrix with planted DE proteins.

    Returns
    -------
    (M_df, design_df, count, is_de)
        M_df       : DataFrame (proteins × samples) of log2 intensities.
        design_df  : DataFrame (samples × groups), one column per group.
        count      : np.ndarray (n_proteins,) peptide counts.
        is_de      : np.ndarray (n_proteins,) bool — ground-truth DE flag.
    """
    rng = np.random.default_rng(seed)
    n_samples = n_per_group * n_groups

    # Per-protein baseline log2 intensity.
    base = rng.normal(20.0, 1.5, size=n_proteins)
    # Per-protein variance, mildly negatively correlated with count.
    count = rng.integers(1, 20, size=n_proteins).astype(float)
    sigma = 0.4 + 1.0 / np.sqrt(count)  # high count ⇒ tighter variance

    # Group means: identical except for DE proteins.
    delta = np.zeros(n_proteins)
    is_de = rng.uniform(size=n_proteins) < prop_de
    delta[is_de] = rng.choice([-1, 1], size=is_de.sum()) * effect_size

    M = np.zeros((n_proteins, n_samples))
    for g in range(n_groups):
        mean_g = base + (delta if g == 1 else 0.0)
        M[:, g * n_per_group:(g + 1) * n_per_group] = (
            rng.normal(loc=mean_g[:, None],
                       scale=sigma[:, None],
                       size=(n_proteins, n_per_group))
        )

    M_df = pd.DataFrame(
        M,
        index=[f"prot_{i:04d}" for i in range(n_proteins)],
        columns=[f"S{j:02d}" for j in range(n_samples)],
    )
    design = np.zeros((n_samples, n_groups))
    for g in range(n_groups):
        design[g * n_per_group:(g + 1) * n_per_group, g] = 1.0
    design_df = pd.DataFrame(
        design,
        index=M_df.columns,
        columns=[f"group{g}" for g in range(n_groups)],
    )
    return M_df, design_df, count, is_de


def test_lmfit_recovers_group_means():
    M, design, _, _ = _make_dataset(seed=0)
    fit = lm_fit(M, design)
    # Coefficients should equal per-group means within ~ noise / sqrt(n_per_group)
    group0_mean = M.iloc[:, :6].mean(axis=1).to_numpy()
    group1_mean = M.iloc[:, 6:].mean(axis=1).to_numpy()
    np.testing.assert_allclose(fit.coefficients[:, 0], group0_mean, atol=1e-8)
    np.testing.assert_allclose(fit.coefficients[:, 1], group1_mean, atol=1e-8)
    assert fit.df_residual[0] == 10  # 12 samples - 2 groups


def test_ebayes_runs_and_produces_p_values():
    M, design, _, is_de = _make_dataset(seed=1)
    # Form a 2-group contrast (group1 vs group0).
    contrast = np.array([-1.0, 1.0])
    fit = lm_fit(M, design)
    from pydeqms.pipeline import _resolve_contrast
    fit, _ = _resolve_contrast(fit, contrast)
    ebayes(fit)
    assert fit.p_value is not None
    assert fit.p_value.shape == (M.shape[0], 1)
    # True DE proteins should have significantly lower p-values.
    p_de = fit.p_value[is_de].mean()
    p_not = fit.p_value[~is_de].mean()
    assert p_de < p_not


def test_spectra_count_ebayes_lower_p_for_high_count_de():
    """When two DE proteins have identical effect but different peptide
    counts, the high-count one should get a lower DEqMS p-value (its
    prior variance is tighter)."""
    M, design, count, is_de = _make_dataset(seed=2, prop_de=0.2)
    contrast = np.array([-1.0, 1.0])
    fit = lm_fit(M, design)
    from pydeqms.pipeline import _resolve_contrast
    fit, _ = _resolve_contrast(fit, contrast)
    spectra_count_ebayes(fit, count=count, fit_method="loess")
    # Mean DEqMS p should be lower for DE proteins.
    sca_p = fit.sca_p[:, 0]
    assert np.nanmean(sca_p[is_de]) < np.nanmean(sca_p[~is_de])
    # And the loess-fit prior variance should be lower for high-count
    # proteins (because high counts mean tight variance in our sim).
    s2p = fit.sca_priorvar
    high_count = count > 15
    low_count = count < 3
    assert np.nanmean(s2p[high_count]) < np.nanmean(s2p[low_count])


def test_deqms_pipeline_returns_sorted_table():
    M, design, count, is_de = _make_dataset(seed=3, prop_de=0.15)
    contrast = np.array([-1.0, 1.0])
    res = deqms(M, design, count=count, contrast=contrast, fit_method="loess")
    # Result should be sorted by sca.P.Value ascending.
    pv = res["sca.P.Value"].to_numpy()
    assert (np.diff(pv) >= -1e-12).all()
    # True DE proteins should populate the top of the ranking.
    top_n = int(is_de.sum())
    top_proteins = set(res["gene"].iloc[:top_n].astype(str))
    true_de_proteins = set(M.index[is_de].astype(str))
    overlap = len(top_proteins & true_de_proteins)
    assert overlap / top_n > 0.5, f"only {overlap}/{top_n} true DE found at top"


def test_deqms_nls_fit_method():
    M, design, count, _ = _make_dataset(seed=4)
    contrast = np.array([-1.0, 1.0])
    res = deqms(M, design, count=count, contrast=contrast, fit_method="nls")
    assert "sca.P.Value" in res.columns


def test_deqms_spline_fit_method():
    M, design, count, _ = _make_dataset(seed=5)
    contrast = np.array([-1.0, 1.0])
    res = deqms(M, design, count=count, contrast=contrast, fit_method="spline")
    assert "sca.P.Value" in res.columns


def test_deqms_handles_missing_values():
    M, design, count, _ = _make_dataset(seed=6)
    M.iloc[0:50, 0] = np.nan          # missing one sample for 50 proteins
    M.iloc[100:110, [0, 1, 2]] = np.nan  # missing 3 samples for 10 proteins
    res = deqms(M, design, count=count, contrast=np.array([-1.0, 1.0]),
                fit_method="loess")
    # All proteins still represented in the result table.
    assert len(res) == M.shape[0]
