"""Head-to-head speed+accuracy benchmark: R limma+DEqMS vs pydeqms.

Runs the full ``lmFit → contrasts.fit → eBayes → spectraCounteBayes``
pipeline on a synthetic 5000-protein × 12-sample TMT-like dataset and
reports:

* wall-clock time for each step (R via ``Rscript``, Python via ``time.perf_counter``)
* MAE / Pearson r between Python and R outputs for: ``lmFit coefs``,
  ``eBayes t``, ``DEqMS sca.t``, ``DEqMS sca.P.Value``

The synthetic matrix is generated deterministically in NumPy and dumped
to TSV so both languages see identical input.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pydeqms import deqms


HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _make_dataset(n_proteins: int, n_per_group: int, n_groups: int = 2, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_samples = n_per_group * n_groups
    base = rng.normal(20.0, 1.5, size=n_proteins)
    count = rng.integers(1, 30, size=n_proteins).astype(float)
    sigma = 0.4 + 1.0 / np.sqrt(count)
    delta = np.zeros(n_proteins)
    is_de = rng.uniform(size=n_proteins) < 0.1
    delta[is_de] = rng.choice([-1, 1], size=is_de.sum()) * 1.0

    M = np.zeros((n_proteins, n_samples))
    for g in range(n_groups):
        mean_g = base + (delta if g == 1 else 0.0)
        M[:, g * n_per_group:(g + 1) * n_per_group] = rng.normal(
            mean_g[:, None], sigma[:, None], size=(n_proteins, n_per_group),
        )
    M_df = pd.DataFrame(
        M,
        index=[f"prot_{i:05d}" for i in range(n_proteins)],
        columns=[f"S{j:02d}" for j in range(n_samples)],
    )
    design = np.zeros((n_samples, n_groups))
    for g in range(n_groups):
        design[g * n_per_group:(g + 1) * n_per_group, g] = 1.0
    design_df = pd.DataFrame(
        design, index=M_df.columns,
        columns=[f"group{g}" for g in range(n_groups)],
    )
    count_s = pd.Series(count, index=M_df.index, name="count")
    return M_df, design_df, count_s, is_de


def run_python(M_df, design_df, count, runs):
    elapsed = []
    for _ in range(runs):
        t0 = time.perf_counter()
        res = deqms(M_df, design_df, count=count,
                    contrast=np.array([-1.0, 1.0]), fit_method="loess")
        elapsed.append(time.perf_counter() - t0)
    return float(np.mean(elapsed)), res


def run_R(work):
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} && Rscript "
        f"{Path(__file__).resolve().parent.parent / 'tests' / 'r_reference_driver.R'} "
        f"{work / 'M.tsv'} {work / 'design.tsv'} {work / 'count.tsv'} "
        f"{work / 'R_out'} loess"
    )
    t0 = time.perf_counter()
    res = subprocess.run(["bash", "-lc", cmd], check=True,
                         capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    timing = pd.read_csv(work / "R_out" / "timing.tsv", sep="\t")
    return elapsed, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-proteins", type=int, default=5000)
    ap.add_argument("--n-per-group", type=int, default=6)
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    M_df, design_df, count, is_de = _make_dataset(
        args.n_proteins, args.n_per_group,
    )
    M_df.to_csv(WORK / "M.tsv", sep="\t", na_rep="NA",
                index_label="protein")
    design_df.to_csv(WORK / "design.tsv", sep="\t", index_label="sample")
    count.to_csv(WORK / "count.tsv", sep="\t", index_label="protein",
                 header=["count"])

    print(f"Matrix: {M_df.shape[0]} proteins × {M_df.shape[1]} samples")

    print(f"\n--- Python timing (mean of {args.runs} runs) ---")
    py_total = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        res_py = deqms(M_df, design_df, count=count,
                       contrast=np.array([-1.0, 1.0]), fit_method="loess")
        py_total.append(time.perf_counter() - t0)
    py_total_mean = float(np.mean(py_total))
    print(f"  py total           {py_total_mean*1000:8.2f} ms")

    print(f"\n--- R timing (single run; per-step breakdown) ---")
    r_total, r_timing = run_R(WORK)
    print(r_timing.to_string(index=False))
    print(f"  R total wall       {r_total*1000:8.2f} ms (including Rscript startup)")

    r_pipeline = float(r_timing.sum(axis=1).iloc[0])
    print(f"\nSpeedup (R pipeline / Python pipeline): {r_pipeline / py_total_mean:.2f}×")

    # Accuracy
    r_sca_t = pd.read_csv(WORK / "R_out" / "sca_t.tsv", sep="\t",
                          index_col=0).iloc[:, 0]
    r_sca_p = pd.read_csv(WORK / "R_out" / "sca_p.tsv", sep="\t",
                          index_col=0).iloc[:, 0]
    py_sca_t = res_py.set_index("gene")["sca.t"].reindex(r_sca_t.index)
    py_sca_p = res_py.set_index("gene")["sca.P.Value"].reindex(r_sca_p.index)

    sca_t_mae = float(np.nanmean(np.abs(py_sca_t.to_numpy() - r_sca_t.to_numpy())))
    sca_t_r = float(np.corrcoef(py_sca_t.dropna(), r_sca_t.loc[py_sca_t.dropna().index])[0, 1])
    sca_p_mae = float(np.nanmean(np.abs(py_sca_p.to_numpy() - r_sca_p.to_numpy())))
    sca_p_r = float(np.corrcoef(py_sca_p.dropna(), r_sca_p.loc[py_sca_p.dropna().index])[0, 1])

    # Top-N gene-set agreement: 10% true-DE proportion → 500 of 5000
    top_n = max(int(args.n_proteins * 0.1), 50)
    r_top = set(r_sca_p.sort_values().index[:top_n])
    py_top = set(res_py.head(top_n)["gene"])
    overlap = len(r_top & py_top) / top_n

    print(f"\n--- Accuracy (Python vs R) ---")
    print(f"  sca.t  MAE={sca_t_mae:.4g}  Pearson r={sca_t_r:.4f}")
    print(f"  sca.P  MAE={sca_p_mae:.4g}  Pearson r={sca_p_r:.4f}")
    print(f"  top-{top_n} agreement: {overlap:.1%}")

    summary = {
        "shape": M_df.shape,
        "py_time_ms": py_total_mean * 1000,
        "r_time_ms_pipeline": r_pipeline * 1000,
        "r_time_ms_total": r_total * 1000,
        "speedup": r_pipeline / py_total_mean,
        "sca_t_mae": sca_t_mae,
        "sca_t_pearson_r": sca_t_r,
        "sca_p_mae": sca_p_mae,
        "sca_p_pearson_r": sca_p_r,
        "top_overlap": overlap,
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report → {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
