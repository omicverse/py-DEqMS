"""Tests for peptide → protein summarization.

Smoke tests on synthetic input plus R-parity checks (Pearson r > 0.999)
against ``DEqMS::medianSummary`` / ``medianSweeping`` / ``medpolishSummary``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from pydeqms import median_summary, median_sweeping, medpolish_summary
from pydeqms.summarization import _median_polish


HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not Path(R_DRIVER).exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(DEqMS); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


def _build_peptide_table(seed: int = 0, n_psms: int = 600, n_proteins: int = 80,
                         n_channels: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    proteins = [f"prot_{i:04d}" for i in range(n_proteins)]
    psm_to_prot = rng.choice(proteins, size=n_psms)
    base = rng.normal(20.0, 2.0, size=n_psms)
    chan_offsets = rng.normal(0.0, 0.5, size=n_channels)
    noise = rng.normal(0.0, 0.4, size=(n_psms, n_channels))
    intensities = base[:, None] + chan_offsets[None, :] + noise
    peptides = [f"pep_{i:05d}" for i in range(n_psms)]
    df = pd.DataFrame(intensities,
                      columns=[f"S{c+1:02d}" for c in range(n_channels)])
    df.insert(0, "protein", psm_to_prot)
    df.insert(0, "peptide", peptides)
    return df


def test_median_summary_smoke():
    df = _build_peptide_table(seed=0)
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    out = median_summary(df, "protein", sample_cols)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == len(sample_cols)
    # One row per unique protein.
    assert len(out) == df["protein"].nunique()


def test_median_summary_with_ref_col():
    df = _build_peptide_table(seed=1)
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    out = median_summary(df, "protein", sample_cols, ref_col="S01")
    # Reference channel should now have median ~0 per protein (since each
    # PSM's reference value is subtracted from itself).
    assert isinstance(out, pd.DataFrame)
    # Reference column should be all near zero.
    np.testing.assert_allclose(out["S01"].to_numpy(),
                               np.zeros(len(out)), atol=1e-12)


def test_median_sweeping_smoke():
    df = _build_peptide_table(seed=2)
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    out = median_sweeping(df, "protein", sample_cols)
    assert out.shape[1] == len(sample_cols)
    # After medianSweeping the column medians should be ~0
    # (equalMedianNormalization is applied at the end).
    np.testing.assert_allclose(out.median(axis=0).to_numpy(),
                               np.zeros(out.shape[1]), atol=1e-12)


def test_medpolish_summary_smoke():
    df = _build_peptide_table(seed=3)
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    out = medpolish_summary(df, "protein", sample_cols)
    assert out.shape[1] == len(sample_cols)


def test_median_polish_matches_known():
    # 3x3 matrix with known row+col additive structure.
    rng = np.random.default_rng(0)
    row = np.array([0.0, 1.0, 2.0])
    col = np.array([0.0, 0.5, -0.5])
    m = row[:, None] + col[None, :]
    res = _median_polish(m)
    # Residuals should all be zero, overall+row+col reproduce m.
    assert np.allclose(res["residuals"], 0.0, atol=1e-10)


@pytest.mark.skipif(not _r_available(), reason="CMAP R env or DEqMS not available.")
def test_median_summary_parity_vs_R(tmp_path):
    df = _build_peptide_table(seed=11, n_psms=400, n_proteins=50, n_channels=10)
    pep_tsv = tmp_path / "peptides.tsv"
    df.to_csv(pep_tsv, sep="\t", index=False, na_rep="NA")
    out_dir = tmp_path / "R_out"
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} summarize {pep_tsv} {out_dir}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)

    r_ms = pd.read_csv(out_dir / "median_summary.tsv", sep="\t", index_col=0)
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    # R uses ref_col=3 → the first sample column.
    py_ms = median_summary(df, "protein", sample_cols, ref_col=sample_cols[0])
    py_aligned = py_ms.reindex(index=r_ms.index, columns=r_ms.columns)

    # Bit-exact: both reduce to colMedians of (sample - ref_col).
    np.testing.assert_allclose(py_aligned.to_numpy(), r_ms.to_numpy(),
                               atol=1e-9, rtol=0,
                               err_msg="medianSummary diverges from R")


@pytest.mark.skipif(not _r_available(), reason="CMAP R env or DEqMS not available.")
def test_median_sweeping_parity_vs_R(tmp_path):
    df = _build_peptide_table(seed=22, n_psms=400, n_proteins=50, n_channels=10)
    pep_tsv = tmp_path / "peptides.tsv"
    df.to_csv(pep_tsv, sep="\t", index=False, na_rep="NA")
    out_dir = tmp_path / "R_out"
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} summarize {pep_tsv} {out_dir}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)
    r_msw = pd.read_csv(out_dir / "median_sweeping.tsv", sep="\t", index_col=0)
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    py_msw = median_sweeping(df, "protein", sample_cols)
    py_aligned = py_msw.reindex(index=r_msw.index, columns=r_msw.columns)

    # Bit-exact target.
    np.testing.assert_allclose(py_aligned.to_numpy(), r_msw.to_numpy(),
                               atol=1e-9, rtol=0,
                               err_msg="medianSweeping diverges from R")


@pytest.mark.skipif(not _r_available(), reason="CMAP R env or DEqMS not available.")
def test_medpolish_summary_parity_vs_R(tmp_path):
    df = _build_peptide_table(seed=33, n_psms=400, n_proteins=50, n_channels=10)
    pep_tsv = tmp_path / "peptides.tsv"
    df.to_csv(pep_tsv, sep="\t", index=False, na_rep="NA")
    out_dir = tmp_path / "R_out"
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} summarize {pep_tsv} {out_dir}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)
    r_mp = pd.read_csv(out_dir / "medpolish.tsv", sep="\t", index_col=0)
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    py_mp = medpolish_summary(df, "protein", sample_cols)
    py_aligned = py_mp.reindex(index=r_mp.index, columns=r_mp.columns)

    # Flatten and check Pearson r > 0.999 (matches R's iterative
    # convergence to within FP).
    r_flat = r_mp.to_numpy().ravel()
    py_flat = py_aligned.to_numpy().ravel()
    mask = np.isfinite(r_flat) & np.isfinite(py_flat)
    r_val, _ = pearsonr(r_flat[mask], py_flat[mask])
    assert r_val > 0.999, (
        f"medpolishSummary Pearson r = {r_val:.6f} (expected > 0.999)"
    )
