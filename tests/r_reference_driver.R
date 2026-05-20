#!/usr/bin/env Rscript
# Drive limma + DEqMS reference on TSV inputs dumped by the Python side.
#
# Usage:
#   Rscript r_reference_driver.R <M_tsv> <design_tsv> <count_tsv> <out_dir> [fit_method]
#
# Outputs (in out_dir):
#   lm_coefs.tsv     limma::lmFit coefficients
#   lm_sigma.tsv     fit$sigma vector
#   lm_su.tsv        fit$stdev.unscaled
#   eb_t.tsv         eBayes moderated t (after contrasts.fit -1/+1)
#   eb_p.tsv         eBayes p
#   eb_meta.tsv      df.prior / s2.prior / df.total[1] / s2.post[1]
#   sca_t.tsv        DEqMS sca.t
#   sca_p.tsv        DEqMS sca.p
#   sca_postvar.tsv  DEqMS sca.postvar
#   sca_priorvar.tsv DEqMS sca.priorvar
#   sca_meta.tsv     sca.dfprior (scalar)
#   timing.tsv       wall-clock for lmFit / eBayes / spectraCounteBayes

suppressPackageStartupMessages({
  library(limma)
  library(DEqMS)
})

args <- commandArgs(trailingOnly = TRUE)
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
