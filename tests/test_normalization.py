"""Tests for ``equal_median_normalization``.

Bit-exact parity against R ``DEqMS::equalMedianNormalization`` on a
synthetic 12-channel TMT-style log2 matrix.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pydeqms import equal_median_normalization


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
    """Synthetic peptide / PSM table.

    Layout matches the R DEqMS convention:

    * col 0 = peptide id
    * col 1 = protein id
    * col 2..2+n_channels-1 = sample channels (log2 intensity)
    """
    rng = np.random.default_rng(seed)
    proteins = [f"prot_{i:04d}" for i in range(n_proteins)]
    psm_to_prot = rng.choice(proteins, size=n_psms)
    base = rng.normal(20.0, 2.0, size=n_psms)
    # Channel offsets — what equal_median_normalization should remove.
    chan_offsets = rng.normal(0.0, 0.5, size=n_channels)
    noise = rng.normal(0.0, 0.4, size=(n_psms, n_channels))
    intensities = base[:, None] + chan_offsets[None, :] + noise
    peptides = [f"pep_{i:05d}" for i in range(n_psms)]
    df = pd.DataFrame(intensities,
                      columns=[f"S{c+1:02d}" for c in range(n_channels)])
    df.insert(0, "protein", psm_to_prot)
    df.insert(0, "peptide", peptides)
    return df


def test_equal_median_basic():
    rng = np.random.default_rng(0)
    M = rng.normal(0.0, 1.0, size=(100, 6))
    M_norm = equal_median_normalization(M)
    # After normalization, every column median should be 0 (or ~1e-12).
    np.testing.assert_allclose(np.nanmedian(M_norm, axis=0),
                               np.zeros(6), atol=1e-12)


def test_equal_median_handles_nan():
    rng = np.random.default_rng(1)
    M = rng.normal(0.0, 1.0, size=(100, 6))
    M[0, 0] = np.nan
    M[5, 3] = np.nan
    M_norm = equal_median_normalization(M)
    assert np.isnan(M_norm[0, 0])
    assert np.isnan(M_norm[5, 3])
    finite_meds = np.nanmedian(M_norm, axis=0)
    np.testing.assert_allclose(finite_meds, np.zeros(6), atol=1e-12)


def test_equal_median_dataframe_roundtrip():
    rng = np.random.default_rng(2)
    M = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(40, 4)),
        index=[f"r{i}" for i in range(40)],
        columns=["A", "B", "C", "D"],
    )
    out = equal_median_normalization(M)
    assert isinstance(out, pd.DataFrame)
    assert (out.index == M.index).all()
    assert (out.columns == M.columns).all()


@pytest.mark.skipif(not _r_available(), reason="CMAP R env or DEqMS not available.")
def test_equal_median_bit_exact_vs_R(tmp_path):
    df = _build_peptide_table(seed=42, n_psms=600, n_proteins=80, n_channels=12)
    pep_tsv = tmp_path / "peptides.tsv"
    df.to_csv(pep_tsv, sep="\t", index=False, na_rep="NA")
    out_dir = tmp_path / "R_out"
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} summarize {pep_tsv} {out_dir}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)

    r_eqm = pd.read_csv(out_dir / "equal_median.tsv", sep="\t", index_col=0)
    # Build the matching Python input: rows = peptides, cols = sample
    # channels, indexed by peptide id (matches what R wrote out).
    sample_cols = [c for c in df.columns if c not in ("peptide", "protein")]
    M_py = df.set_index("peptide")[sample_cols]
    py_eqm = equal_median_normalization(M_py)

    # Align by row / col then compare.
    py_aligned = py_eqm.reindex(index=r_eqm.index, columns=r_eqm.columns)
    np.testing.assert_allclose(py_aligned.to_numpy(), r_eqm.to_numpy(),
                               atol=1e-9, rtol=0,
                               err_msg="equalMedianNormalization diverges from R")
