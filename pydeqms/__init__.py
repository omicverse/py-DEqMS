"""
pydeqms: Pure-Python port of Bioconductor DEqMS.

Peptide-count-aware moderated t-tests for differential mass-spec
proteomics (Zhu et al. 2020, MCP). Three-step core pipeline mirroring
the R reference:

* :func:`lm_fit` — per-protein OLS (``limma::lmFit`` minimal).
* :func:`ebayes` — Smyth 2004 empirical-Bayes moderated t (``limma::eBayes``).
* :func:`spectra_count_ebayes` — DEqMS posterior with a loess/nls/spline
  fit of ``log(s²) ~ log2(count)`` (``DEqMS::spectraCounteBayes``).

Peptide → protein summarization (TMT / iTRAQ / LFQ):

* :func:`equal_median_normalization` — ``DEqMS::equalMedianNormalization``.
* :func:`median_summary` — ``DEqMS::medianSummary``.
* :func:`median_sweeping` — ``DEqMS::medianSweeping``.
* :func:`medpolish_summary` — ``DEqMS::medpolishSummary``.

Plotting (lazy-imported matplotlib):

* :func:`peptide_profile_plot` — ``DEqMS::peptideProfilePlot``.
* :func:`residual_plot` — ``DEqMS::Residualplot``.
* :func:`variance_boxplot` — ``DEqMS::VarianceBoxplot``.
* :func:`variance_scatter_plot` — ``DEqMS::VarianceScatterplot``.

A convenience :func:`deqms` wraps the three core steps and returns the
topTable-style DataFrame (``DEqMS::outputResult``).

Quick-start
-----------
>>> from pydeqms import deqms
>>> # M : log2 protein intensity matrix (proteins × samples) as DataFrame
>>> # design : np.ndarray (samples × groups)
>>> # count : peptide-count vector (length = n_proteins)
>>> # contrast : np.ndarray (groups,) or index into design
>>> result = deqms(M, design, count=count, contrast=contrast)
>>> result.head()
"""
from __future__ import annotations

from .deqms import output_result, spectra_count_ebayes
from .ebayes import ebayes
from .lmfit import LinearModelFit, lm_fit
from .normalization import equal_median_normalization
from .pipeline import deqms
from .plotting import (
    peptide_profile_plot,
    residual_plot,
    variance_boxplot,
    variance_scatter_plot,
)
from .summarization import (
    median_summary,
    median_sweeping,
    medpolish_summary,
)

__version__ = "0.1.1"

__all__ = [
    "deqms",
    "ebayes",
    "lm_fit",
    "output_result",
    "spectra_count_ebayes",
    "LinearModelFit",
    "equal_median_normalization",
    "median_summary",
    "median_sweeping",
    "medpolish_summary",
    "peptide_profile_plot",
    "residual_plot",
    "variance_boxplot",
    "variance_scatter_plot",
]
