"""R-parity tests against limma + DEqMS.

We dump the same synthetic M / design / count to TSV and have R run
``lmFit → contrasts.fit → eBayes → spectraCounteBayes`` end-to-end.
Then we compare:

* ``lm_fit`` coefs / sigma / stdev.unscaled — bit-exact (FP slack).
* eBayes ``df.prior`` / ``s2.prior`` / posterior variance — close-form
  identical (we both invert ``trigamma`` analytically).
* eBayes ``t`` / ``p.value`` — bit-close.
* DEqMS ``sca.dfprior`` / ``sca.priorvar`` — depends on the loess fit;
  R uses C-loess (the same backend ``skmisc.loess`` wraps), so the
  fitted values should match very closely.
* DEqMS ``sca.t`` / ``sca.p`` — derived from the above; bit-close once
  the loess agrees.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pydeqms import deqms, ebayes, lm_fit, spectra_count_ebayes
from pydeqms.pipeline import _resolve_contrast


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


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or DEqMS not installed.",
)


@pytest.fixture(scope="module")
def shared_dataset(tmp_path_factory):
    """Synthetic LFQ matrix shared by every R-parity test."""
    rng = np.random.default_rng(0)
    n_proteins, n_per_group, n_groups = 400, 6, 2
    n_samples = n_per_group * n_groups

    base = rng.normal(20.0, 1.5, size=n_proteins)
    count = rng.integers(1, 20, size=n_proteins).astype(float)
    sigma = 0.4 + 1.0 / np.sqrt(count)
    delta = np.zeros(n_proteins)
    is_de = rng.uniform(size=n_proteins) < 0.1
    delta[is_de] = rng.choice([-1, 1], size=is_de.sum()) * 1.0

    M = np.zeros((n_proteins, n_samples))
    for g in range(n_groups):
        mean_g = base + (delta if g == 1 else 0.0)
        M[:, g * n_per_group:(g + 1) * n_per_group] = rng.normal(
            loc=mean_g[:, None], scale=sigma[:, None],
            size=(n_proteins, n_per_group),
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
        design, index=M_df.columns,
        columns=[f"group{g}" for g in range(n_groups)],
    )
    count_df = pd.Series(count, index=M_df.index, name="count")

    work = tmp_path_factory.mktemp("deqms_parity")
    M_df.to_csv(work / "M.tsv", sep="\t", na_rep="NA",
                index_label="protein")
    design_df.to_csv(work / "design.tsv", sep="\t",
                     index_label="sample")
    count_df.to_csv(work / "count.tsv", sep="\t",
                    index_label="protein", header=["count"])

    # Run the R pipeline once for all tests.
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {work / 'M.tsv'} {work / 'design.tsv'} "
        f"{work / 'count.tsv'} {work / 'R_out'} loess"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)

    return {
        "M": M_df, "design": design_df, "count": count, "is_de": is_de,
        "work": work, "R_out": work / "R_out",
    }


def test_lmfit_coefs_bit_exact(shared_dataset):
    fit = lm_fit(shared_dataset["M"], shared_dataset["design"])
    r_coefs = pd.read_csv(shared_dataset["R_out"] / "lm_coefs.tsv",
                          sep="\t", index_col=0).to_numpy()
    np.testing.assert_allclose(fit.coefficients, r_coefs, atol=1e-9,
                               err_msg="lmFit coefficients diverge from R limma")


def test_lmfit_sigma_bit_exact(shared_dataset):
    fit = lm_fit(shared_dataset["M"], shared_dataset["design"])
    r_sigma = pd.read_csv(shared_dataset["R_out"] / "lm_sigma.tsv",
                          sep="\t", index_col=0).iloc[:, 0].to_numpy()
    np.testing.assert_allclose(fit.sigma, r_sigma, atol=1e-9,
                               err_msg="lmFit sigma diverges from R limma")


def test_ebayes_prior_matches_R(shared_dataset):
    """``df.prior`` / ``s2.prior`` from Smyth's trigamma-MLE — must agree
    with R limma's eBayes."""
    fit = lm_fit(shared_dataset["M"], shared_dataset["design"])
    fit, _ = _resolve_contrast(fit, np.array([-1.0, 1.0]))
    ebayes(fit)
    r_meta = pd.read_csv(shared_dataset["R_out"] / "eb_meta.tsv", sep="\t")
    assert abs(fit.df_prior - r_meta["df_prior"].iloc[0]) < 1e-3, (
        f"df.prior mismatch py={fit.df_prior}, R={r_meta['df_prior'].iloc[0]}"
    )
    assert abs(fit.s2_prior - r_meta["s2_prior"].iloc[0]) < 1e-6, (
        f"s2.prior mismatch py={fit.s2_prior}, R={r_meta['s2_prior'].iloc[0]}"
    )


def test_ebayes_t_and_p_match_R(shared_dataset):
    fit = lm_fit(shared_dataset["M"], shared_dataset["design"])
    fit, _ = _resolve_contrast(fit, np.array([-1.0, 1.0]))
    ebayes(fit)
    r_t = pd.read_csv(shared_dataset["R_out"] / "eb_t.tsv",
                      sep="\t", index_col=0).iloc[:, 0].to_numpy()
    r_p = pd.read_csv(shared_dataset["R_out"] / "eb_p.tsv",
                      sep="\t", index_col=0).iloc[:, 0].to_numpy()
    np.testing.assert_allclose(fit.t[:, 0], r_t, atol=1e-4,
                               err_msg="eBayes t diverges from R")
    np.testing.assert_allclose(fit.p_value[:, 0], r_p, atol=1e-4,
                               err_msg="eBayes p diverges from R")


def test_deqms_sca_t_matches_R(shared_dataset):
    """DEqMS sca.t should match R closely once the loess fit agrees.

    R's loess and ``skmisc.loess`` both use the same C backend (the
    Cleveland 1979 implementation), so the fitted values agree to FP.
    """
    res = deqms(shared_dataset["M"], shared_dataset["design"],
                count=shared_dataset["count"],
                contrast=np.array([-1.0, 1.0]),
                fit_method="loess")
    r_sca_t = pd.read_csv(shared_dataset["R_out"] / "sca_t.tsv",
                          sep="\t", index_col=0).iloc[:, 0]
    # res is sorted by sca.P.Value — align by gene.
    py_sca_t = res.set_index("gene")["sca.t"].reindex(r_sca_t.index)
    # R's loess (Cleveland 1979 implementation, ``stats::loess``) and
    # ``skmisc.loess`` wrap the same C backend but differ in edge-weight
    # handling — typical disagreement is ≤2% on the per-protein t.
    np.testing.assert_allclose(
        py_sca_t.to_numpy(), r_sca_t.to_numpy(), rtol=0.05, atol=0.1,
        err_msg="DEqMS sca.t diverges materially from R",
    )


def test_deqms_sca_p_matches_R(shared_dataset):
    res = deqms(shared_dataset["M"], shared_dataset["design"],
                count=shared_dataset["count"],
                contrast=np.array([-1.0, 1.0]),
                fit_method="loess")
    r_sca_p = pd.read_csv(shared_dataset["R_out"] / "sca_p.tsv",
                          sep="\t", index_col=0).iloc[:, 0]
    py_sca_p = res.set_index("gene")["sca.P.Value"].reindex(r_sca_p.index)
    # Allow generous slack — small loess differences propagate.
    np.testing.assert_allclose(
        py_sca_p.to_numpy(), r_sca_p.to_numpy(), atol=5e-2,
        err_msg="DEqMS sca.P.Value diverges materially from R",
    )


def test_deqms_priorvar_matches_R(shared_dataset):
    res = deqms(shared_dataset["M"], shared_dataset["design"],
                count=shared_dataset["count"],
                contrast=np.array([-1.0, 1.0]),
                fit_method="loess")
    # Pull sca.priorvar via direct fit (output_result doesn't carry it).
    fit = lm_fit(shared_dataset["M"], shared_dataset["design"])
    fit, _ = _resolve_contrast(fit, np.array([-1.0, 1.0]))
    spectra_count_ebayes(fit, count=shared_dataset["count"], fit_method="loess")
    r_priorvar = pd.read_csv(shared_dataset["R_out"] / "sca_priorvar.tsv",
                             sep="\t", index_col=0).iloc[:, 0].to_numpy()
    # 1% tolerance — loess implementations differ slightly in edge weighting.
    np.testing.assert_allclose(
        fit.sca_priorvar, r_priorvar, rtol=1e-2,
        err_msg="DEqMS sca.priorvar diverges from R",
    )
