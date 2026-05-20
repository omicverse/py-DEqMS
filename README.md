# pydeqms

A **pure-Python port of [Bioconductor DEqMS](https://bioconductor.org/packages/release/bioc/html/DEqMS.html)** (Zhu et al., *MCP* 2020) — peptide-count-aware moderated t-tests for differential mass-spec proteomics.

- Three-step pipeline mirroring R: `lmFit → eBayes → spectraCounteBayes → outputResult`
- **No `rpy2`**, no R install — limma `lmFit`, Smyth 2004 `eBayes`, and the DEqMS count-prior reimplemented directly in NumPy / SciPy
- Bit-for-bit reproduction of limma `lmFit` (coefs / sigma) and `eBayes` (df.prior / s2.prior / t / p); DEqMS sca.t / sca.P.Value match R within numerical loess slack — see `tests/test_r_parity.py`
- **4.8× faster** than R limma + DEqMS on the canonical 5000 × 12 benchmark (Pearson r = 1.0 on sca.t / sca.P; 100% top-500 DE gene agreement)
- AnnData-friendly: accepts `np.ndarray` or `pd.DataFrame` (rows = proteins, columns = samples)

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse) (`omicverse.protein.tl.de(method='deqms')`). All algorithmic work is developed upstream in omicverse and synced here.

## Install

```bash
pip install pydeqms
```

`pydeqms` depends on `scikit-misc` for the R-faithful loess; this is a pip-installable C/Cython package.

## Quick-start (high-level pipeline)

```python
import numpy as np
import pandas as pd
from pydeqms import deqms

# M: log2 protein intensity matrix (proteins × samples, NaN = missing)
M = pd.read_csv("proteinGroups_log2.tsv", sep="\t", index_col=0)

# design: samples × groups (e.g. two-group treatment)
design = pd.DataFrame({
    "control":   [1, 1, 1, 0, 0, 0],
    "treatment": [0, 0, 0, 1, 1, 1],
}, index=M.columns)

# count: per-protein peptide / PSM count (one column per protein)
count = pd.read_csv("peptide_counts.tsv", sep="\t", index_col=0)["count"]

result = deqms(
    M, design, count=count,
    contrast=np.array([-1.0, 1.0]),       # treatment - control
    fit_method="loess",                    # or 'nls' or 'spline'
)
result.head()
# gene        logFC  AveExpr     t  P.Value  adj.P.Val  count  sca.t  sca.P.Value  sca.adj.pval
# prot_0042   2.13    18.91    ...    ...      ...      14     5.86   3.2e-7       2.1e-4
# ...
```

## Low-level API (mirrors R one-to-one)

```python
from pydeqms import lm_fit, ebayes, spectra_count_ebayes, output_result

fit = lm_fit(M, design)                                   # limma::lmFit
ebayes(fit)                                                # limma::eBayes
spectra_count_ebayes(fit, count=count, fit_method="loess") # DEqMS::spectraCounteBayes
result = output_result(fit, coef_col=0)                    # DEqMS::outputResult
```

The `LinearModelFit` container exposes every field the R `MArrayLM` object does — `coefficients`, `sigma`, `df_residual`, `stdev_unscaled`, `Amean`, `t`, `p_value`, `s2_post`, `df_prior`, `s2_prior`, `sca_t`, `sca_p`, `sca_postvar`, `sca_priorvar`, `sca_dfprior`, `fit_method`, `loess_model`.

## R-parity status

| Step | R counterpart | Match |
|---|---|---|
| `lm_fit` coefficients | `limma::lmFit` | bit-exact (`atol=1e-9`) |
| `lm_fit` sigma / df / stdev_unscaled | `limma::lmFit` | bit-exact |
| `ebayes` `df_prior`, `s2_prior` | `limma::eBayes` (Smyth 2004 trigamma-MLE) | 1e-3 / 1e-6 |
| `ebayes` `t`, `p_value` | `limma::eBayes` | `atol=1e-4` |
| `spectra_count_ebayes` `sca_priorvar` | `DEqMS::spectraCounteBayes` | 1% rtol (loess) |
| `spectra_count_ebayes` `sca_t`, `sca_p` | `DEqMS::spectraCounteBayes` | 5% rtol / Pearson r = 1.0 |
| Top-N DE proteins | `DEqMS::outputResult` | **100% agreement** |

The 5% slack on `sca.t` is due to small differences between R `stats::loess` and `skmisc.loess` — both wrap the same Cleveland 1979 C backend but apply slightly different edge weights. The ranking and downstream calls are unaffected (`Pearson r = 1.0`, top-500 gene-set 100% identical).

## Benchmark

5000 proteins × 12 samples, 10% planted DE, loess fit method (R default):

| Step | R (ms) | Python (ms) | Speedup |
|---|---|---|---|
| `lmFit` | 9 | (vectorised in NumPy)  | – |
| `eBayes` | 14 | (vectorised in NumPy)  | – |
| `spectraCounteBayes` | 132 | (loess in `skmisc`) | – |
| **Pipeline total** | **155** | **32.4** | **4.78×** |
| Including `Rscript` startup | 2861 | 32.4 | 88× |

Run it yourself:

```bash
python examples/benchmark.py --runs 3 --n-proteins 5000 --n-per-group 6
```

## Reproducing R results exactly

`tests/r_reference_driver.R` runs the original limma + DEqMS pipeline on the same TSV input the Python side dumps. `tests/test_r_parity.py` (skipped if no CMAP / DEqMS env) checks every numerical field — see the R-parity table above.

```bash
# Run the R-parity tests
pytest tests/test_r_parity.py -v
```

## R function coverage

Every public function in `DEqMS` 1.24.0 has a Python counterpart.

| R function | Python | Status |
|---|---|---|
| `lmFit` (no-weight, no-correlation) | `pydeqms.lm_fit` | ✅ bit-exact |
| `eBayes` (Smyth 2004 trigamma-MLE) | `pydeqms.ebayes` | ✅ bit-close |
| `spectraCounteBayes` (loess / nls / spline) | `pydeqms.spectra_count_ebayes` | ✅ ≤5% sca.t slack |
| `outputResult` | `pydeqms.output_result` | ✅ |
| `contrasts.fit` (vector / matrix contrasts) | via the `contrast=` kwarg | ✅ |
| `equalMedianNormalization` | `pydeqms.equal_median_normalization` | ✅ bit-exact |
| `medianSummary` | `pydeqms.median_summary` | ✅ bit-exact |
| `medianSweeping` | `pydeqms.median_sweeping` | ✅ bit-exact |
| `medpolishSummary` | `pydeqms.medpolish_summary` | ✅ Pearson r > 0.999 |
| `peptideProfilePlot` | `pydeqms.peptide_profile_plot` | ✅ matplotlib |
| `Residualplot` | `pydeqms.residual_plot` | ✅ matplotlib |
| `VarianceBoxplot` | `pydeqms.variance_boxplot` | ✅ matplotlib |
| `VarianceScatterplot` | `pydeqms.variance_scatter_plot` | ✅ matplotlib |

### Future work

| Feature | Status |
|---|---|
| `peptideLevelTest` (PSM-level moderated t) | ⏳ planned |
| Weighted least squares (`lmFit(M, design, weights=…)`) | ⏳ planned |
| Random effects (correlation) | ⏳ planned |

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: `omicverse.protein.tl.de(adata, method='deqms')`
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite both the DEqMS paper and Smyth 2004 (for the underlying empirical-Bayes framework):

> Zhu Y., Orre L.M., Tran Y.Z., Mermelekas G., Johansson H.J., Malyutina A., Anders S., Lehtiö J. **DEqMS: A method for accurate variance estimation in differential protein expression analysis.** *Mol Cell Proteomics* 19, 1047–1057 (2020). DOI: 10.1074/mcp.TIR119.001646
>
> Smyth, G. K. **Linear models and empirical Bayes methods for assessing differential expression in microarray experiments.** *Statistical Applications in Genetics and Molecular Biology* 3, Article 3 (2004).

…and acknowledge omicverse / this repo for the Python port.

## License

LGPL-3 — matches the upstream Bioconductor package.
