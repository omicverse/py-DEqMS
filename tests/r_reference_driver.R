#!/usr/bin/env Rscript
# Drive limma + DEqMS reference on TSV inputs dumped by the Python side.
#
# Usage (mode='pipeline', original; arg5 = fit_method):
#   Rscript r_reference_driver.R <M_tsv> <design_tsv> <count_tsv> <out_dir> [fit_method]
#
# Usage (mode='summarize'):
#   Rscript r_reference_driver.R summarize <peptide_tsv> <out_dir>
#
# Outputs (mode='pipeline', in out_dir):
#   lm_coefs.tsv / lm_sigma.tsv / lm_su.tsv
#   eb_t.tsv / eb_p.tsv / eb_meta.tsv
#   sca_t.tsv / sca_p.tsv / sca_postvar.tsv / sca_priorvar.tsv / sca_meta.tsv
#   timing.tsv
#
# Outputs (mode='summarize', in out_dir):
#   equal_median.tsv     equalMedianNormalization(dat[,3:ncol])
#   median_summary.tsv   medianSummary(dat, group_col=2, ref_col=3)
#   median_sweeping.tsv  medianSweeping(dat, group_col=2)
#   medpolish.tsv        medpolishSummary(dat, group_col=2)

suppressPackageStartupMessages({
  library(limma)
  library(DEqMS)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1 && args[[1]] == "summarize") {
  peptide_tsv <- args[[2]]
  out_dir     <- args[[3]]
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  dat <- read.table(peptide_tsv, sep = "\t", header = TRUE,
                    check.names = FALSE, stringsAsFactors = FALSE,
                    na.strings = c("NA", ""))
  # dat layout: col1 = peptide, col2 = protein, col3..N = sample channels.
  sample_cols <- colnames(dat)[3:ncol(dat)]
  # equalMedianNormalization on the sample block (matrix).
  M_sample <- as.matrix(dat[, 3:ncol(dat)])
  rownames(M_sample) <- dat[, 1]
  M_norm <- equalMedianNormalization(M_sample)
  write.table(M_norm, file.path(out_dir, "equal_median.tsv"),
              sep = "\t", quote = FALSE, col.names = NA)

  # medianSummary: use the first sample column as the reference.
  ms <- medianSummary(dat, group_col = 2, ref_col = 3)
  write.table(ms, file.path(out_dir, "median_summary.tsv"),
              sep = "\t", quote = FALSE, col.names = NA)

  msw <- medianSweeping(dat, group_col = 2)
  write.table(msw, file.path(out_dir, "median_sweeping.tsv"),
              sep = "\t", quote = FALSE, col.names = NA)

  mp <- medpolishSummary(dat, group_col = 2)
  write.table(mp, file.path(out_dir, "medpolish.tsv"),
              sep = "\t", quote = FALSE, col.names = NA)

  cat("R summarize done\n")
  quit(status = 0)
}

M_tsv      <- args[[1]]
design_tsv <- args[[2]]
count_tsv  <- args[[3]]
out_dir    <- args[[4]]
fit_method <- if (length(args) >= 5) args[[5]] else "loess"

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

M <- as.matrix(read.table(M_tsv, sep = "\t", header = TRUE, row.names = 1,
                          check.names = FALSE, na.strings = c("NA","")))
design <- as.matrix(read.table(design_tsv, sep = "\t", header = TRUE,
                               row.names = 1, check.names = FALSE))
count <- read.table(count_tsv, sep = "\t", header = TRUE, row.names = 1,
                    check.names = FALSE)
count_vec <- as.numeric(count[, 1])
names(count_vec) <- rownames(count)

# Two-group contrast: group1 - group0. Matches the Python smoke setup.
contrast.matrix <- makeContrasts(group_diff = group1 - group0,
                                 levels = colnames(design))

t0 <- proc.time()[[3]]
fit <- lmFit(M, design)
t1 <- proc.time()[[3]]
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)
t2 <- proc.time()[[3]]
fit2$count <- count_vec
fit3 <- spectraCounteBayes(fit2, fit.method = fit_method, coef_col = 1)
t3 <- proc.time()[[3]]

write.table(fit$coefficients, file.path(out_dir, "lm_coefs.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(fit$sigma, file.path(out_dir, "lm_sigma.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(fit$stdev.unscaled, file.path(out_dir, "lm_su.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

write.table(fit2$t,        file.path(out_dir, "eb_t.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(fit2$p.value,  file.path(out_dir, "eb_p.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(data.frame(df_prior = fit2$df.prior,
                       s2_prior = fit2$s2.prior,
                       df_total_first = fit2$df.total[1],
                       s2_post_first  = fit2$s2.post[1]),
            file.path(out_dir, "eb_meta.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

write.table(fit3$sca.t,        file.path(out_dir, "sca_t.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(fit3$sca.p,        file.path(out_dir, "sca_p.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(fit3$sca.postvar,  file.path(out_dir, "sca_postvar.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(fit3$sca.priorvar, file.path(out_dir, "sca_priorvar.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(data.frame(sca_dfprior = fit3$sca.dfprior),
            file.path(out_dir, "sca_meta.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(lmFit = t1 - t0,
                       eBayes = t2 - t1,
                       spectraCounteBayes = t3 - t2),
            file.path(out_dir, "timing.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("R DEqMS pipeline done\n")
