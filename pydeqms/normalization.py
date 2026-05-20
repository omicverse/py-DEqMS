"""Per-channel normalization for TMT / iTRAQ / LFQ matrices.

Port of ``DEqMS::equalMedianNormalization``. The R source is::

    function (dat) {
        sizefactor = matrixStats::colMedians(as.matrix(dat), na.rm=TRUE)
        dat.nm = sweep(dat, 2, sizefactor)
        return(dat.nm)
    }

i.e. each column has its median subtracted (no overall-median add-back).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_median_normalization(
    M: np.ndarray | pd.DataFrame,
) -> np.ndarray | pd.DataFrame:
    r"""Subtract each column's median.

    Mirrors ``DEqMS::equalMedianNormalization``. Used by TMT / iTRAQ
    workflows to align per-channel medians after log2-transform.

    Parameters
    ----------
    M
        Protein × sample matrix (``np.ndarray`` or ``pd.DataFrame``). NaN
        entries are ignored when computing column medians (matching
        ``matrixStats::colMedians(..., na.rm=TRUE)``).

    Returns
    -------
    Same type as input (DataFrame in → DataFrame out, ndarray → ndarray)
    with each column's median subtracted.
    """
    if isinstance(M, pd.DataFrame):
        arr = M.to_numpy(dtype=float).copy()
        # nanmedian on each column (axis=0).
        col_med = np.nanmedian(arr, axis=0)
        arr -= col_med[None, :]
        return pd.DataFrame(arr, index=M.index, columns=M.columns)
    arr = np.asarray(M, dtype=float).copy()
    col_med = np.nanmedian(arr, axis=0)
    arr -= col_med[None, :]
    return arr
