"""
pydeqms: Pure-Python port of Bioconductor DEqMS.

Peptide-count-aware moderated t-tests for differential mass-spec
proteomics (Zhu et al. 2020, MCP). Three-step pipeline mirroring the
R reference:

* :func:`lm_fit` — per-protein OLS (``limma::lmFit`` minimal).
* :func:`ebayes` — Smyth 2004 empirical-Bayes moderated t (``limma::eBayes``).
* :func:`spectra_count_ebayes` — DEqMS posterior with a loess/nls/spline
  fit of ``log(s²) ~ log2(count)`` (``DEqMS::spectraCounteBayes``).

A convenience :func:`deqms` wraps the three together and returns the
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
from .pipeline import deqms

__version__ = "0.1.0"

__all__ = [
    "deqms",
    "ebayes",
    "lm_fit",
    "output_result",
    "spectra_count_ebayes",
    "LinearModelFit",
]
