"""Peptide → protein summarization (median, median-sweeping, median polish).

Port of:

* ``DEqMS::medianSummary``    — per-protein median across PSMs (with optional
  reference-channel subtraction).
* ``DEqMS::medianSweeping``   — TMT median-sweeping: subtract per-PSM median,
  then take per-protein median, then ``equalMedianNormalization``.
* ``DEqMS::medpolishSummary`` — Tukey median polish, returning per-protein
  column effects (matches R ``medianPolish`` which exposes only ``$col``).

The R API uses a flat peptide table where col 1 = peptide id, col
``group_col`` = protein id (default ``group_col=2``), and the remaining
columns are sample channels. The Python signature is explicit:
``peptide_df, protein_col, sample_cols`` plus the same ``ref_col`` arg.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .normalization import equal_median_normalization


RefColLike = Union[str, Sequence[str], None]


def _coerce_inputs(
    peptide_df: pd.DataFrame,
    protein_col: str,
    sample_cols: Sequence[str],
) -> tuple[pd.Series, np.ndarray]:
    if protein_col not in peptide_df.columns:
        raise KeyError(f"protein_col {protein_col!r} not in peptide_df.columns")
    missing = [c for c in sample_cols if c not in peptide_df.columns]
    if missing:
        raise KeyError(f"sample_cols missing from peptide_df: {missing}")
    proteins = peptide_df[protein_col]
    sample_arr = peptide_df.loc[:, list(sample_cols)].to_numpy(dtype=float)
    return proteins, sample_arr


def median_summary(
    peptide_df: pd.DataFrame,
    protein_col: str,
    sample_cols: Sequence[str],
    ref_col: RefColLike = None,
) -> pd.DataFrame:
    r"""Per-protein **median** across PSMs (``DEqMS::medianSummary``).

    Parameters
    ----------
    peptide_df
        Peptide / PSM table. Must contain ``protein_col`` and every name
        in ``sample_cols``. The DataFrame index / extra columns are
        ignored.
    protein_col
        Column name holding the protein identifier (grouping key).
    sample_cols
        Sample-channel column names to summarize over.
    ref_col
        Optional reference-channel name (``str``) or list of reference
        channels (``Sequence[str]``). If given, every sample column is
        re-expressed as ``sample - mean(ref_col)`` (or
        ``sample - ref_col`` when a single reference). This mirrors R's
        behaviour when ``ref_col`` is supplied.

    Returns
    -------
    pd.DataFrame
        Protein × sample DataFrame, indexed by protein id. NaNs are
        skipped when taking medians.
    """
    proteins, sample_arr = _coerce_inputs(peptide_df, protein_col, sample_cols)

    if ref_col is not None:
        if isinstance(ref_col, str):
            ref = peptide_df[ref_col].to_numpy(dtype=float)
            sample_arr = sample_arr - ref[:, None]
        else:
            ref_list = list(ref_col)
            ref_block = peptide_df.loc[:, ref_list].to_numpy(dtype=float)
            ref_mean = np.nanmean(ref_block, axis=1)
            sample_arr = sample_arr - ref_mean[:, None]

    df = pd.DataFrame(sample_arr, columns=list(sample_cols))
    df["__protein__"] = proteins.to_numpy()
    out = df.groupby("__protein__", sort=True).agg(
        lambda x: np.nanmedian(x.to_numpy(dtype=float))
    )
    out.index.name = protein_col
    return out


def median_sweeping(
    peptide_df: pd.DataFrame,
    protein_col: str,
    sample_cols: Sequence[str],
) -> pd.DataFrame:
    r"""TMT **median-sweeping** summarization (``DEqMS::medianSweeping``).

    Per-PSM, subtract the median across the sample channels; then take
    the per-protein median across PSMs; finally column-median-normalise
    (``equal_median_normalization``).

    Parameters
    ----------
    peptide_df
        Peptide / PSM table.
    protein_col
        Column name holding the protein identifier.
    sample_cols
        Sample-channel column names.

    Returns
    -------
    pd.DataFrame
        Protein × sample, column-median-normalised. Indexed by protein.
    """
    proteins, sample_arr = _coerce_inputs(peptide_df, protein_col, sample_cols)
    # Per-PSM sweep: subtract row-median over the sample channels.
    row_med = np.nanmedian(sample_arr, axis=1)
    swept = sample_arr - row_med[:, None]

    df = pd.DataFrame(swept, columns=list(sample_cols))
    df["__protein__"] = proteins.to_numpy()
    out = df.groupby("__protein__", sort=True).agg(
        lambda x: np.nanmedian(x.to_numpy(dtype=float))
    )
    out.index.name = protein_col
    return equal_median_normalization(out)


def _median_polish(
    m: np.ndarray, eps: float = 0.01, maxiter: int = 10,
) -> dict:
    r"""Tukey median polish — Python port of R's ``stats::medpolish``.

    Iteratively subtracts row and column medians until the residuals
    stabilise. Returns a dict with ``overall``, ``row``, ``col``, and
    final ``residuals``. Matches R defaults (``eps=0.01, maxiter=10``).
    """
    z = np.array(m, dtype=float, copy=True)
    nr, nc = z.shape
    t = 0.0
    r = np.zeros(nr)
    c = np.zeros(nc)
    old_sum = 0.0
    for _ in range(maxiter):
        # 1. row medians of residuals → push into row effect.
        rdelta = np.nanmedian(z, axis=1)
        z = z - rdelta[:, None]
        r = r + rdelta
        # Re-center column effects using the median of `c`.
        delta = np.nanmedian(c)
        c = c - delta
        t = t + delta

        # 2. column medians of residuals → push into col effect.
        cdelta = np.nanmedian(z, axis=0)
        z = z - cdelta[None, :]
        c = c + cdelta
        delta = np.nanmedian(r)
        r = r - delta
        t = t + delta

        new_sum = np.nansum(np.abs(z))
        converged = (
            new_sum == 0
            or abs(new_sum - old_sum) < eps * new_sum
        )
        if converged:
            break
        old_sum = new_sum
    return {"overall": float(t), "row": r, "col": c, "residuals": z}


def medpolish_summary(
    peptide_df: pd.DataFrame,
    protein_col: str,
    sample_cols: Sequence[str],
) -> pd.DataFrame:
    r"""Per-protein **Tukey median polish** summarization (``DEqMS::medpolishSummary``).

    For each protein, fit a row+col median-polish model to its PSM
    submatrix and use the **column effects** as the summarized value.
    This matches the R helper ``medianPolish`` inside
    ``DEqMS::medpolishSummary`` which returns ``medpolish(...)$col``.

    Parameters
    ----------
    peptide_df
        Peptide / PSM table.
    protein_col
        Column name with protein identifier.
    sample_cols
        Sample-channel column names.

    Returns
    -------
    pd.DataFrame
        Protein × sample, indexed by protein. Note: returns the
        median-polish **column** effects, not absolute intensities
        (matches R behavior).
    """
    proteins, sample_arr = _coerce_inputs(peptide_df, protein_col, sample_cols)
    sample_cols = list(sample_cols)
    rows = []
    proteins_arr = proteins.to_numpy()
    unique = pd.unique(proteins_arr)
    # Match R `plyr::ddply` ordering (sorted by group key).
    unique = np.array(sorted(unique, key=lambda v: (str(v))))
    for p in unique:
        mask = proteins_arr == p
        sub = sample_arr[mask, :]
        if sub.shape[0] == 1:
            # medpolish of single row reduces to col residual = 0.
            rows.append(np.zeros(sub.shape[1]))
        else:
            mp = _median_polish(sub)
            rows.append(mp["col"])
    out = pd.DataFrame(np.vstack(rows) if rows else np.empty((0, len(sample_cols))),
                       columns=sample_cols, index=unique)
    out.index.name = protein_col
    return out
