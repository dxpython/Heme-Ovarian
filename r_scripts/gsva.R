args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NA) {
  idx <- grep(paste0("^--", name, "$"), args)
  if (length(idx) == 0) return(default)
  args[idx + 1]
}
expr_path <- get_arg("expr")
gmt_path <- get_arg("gmt")
out_path <- get_arg("out")
kernel <- get_arg("kernel", "gaussian")
if (is.na(expr_path) || is.na(gmt_path) || is.na(out_path)) {
  stop("usage: Rscript gsva.R --expr PATH --gmt PATH --out PATH [--kernel gaussian]")
}
if (!requireNamespace("GSVA", quietly = TRUE)) {
  stop("Bioconductor package GSVA is required")
}
if (!requireNamespace("GSEABase", quietly = TRUE)) {
  stop("Bioconductor package GSEABase is required")
}
expr <- as.matrix(read.csv(expr_path, row.names = 1, check.names = FALSE))
geneset <- GSEABase::getGmt(gmt_path)
param <- GSVA::gsvaParam(exprData = expr, geneSets = geneset, kcdf = if (kernel == "gaussian") "Gaussian" else "Poisson")
res <- GSVA::gsva(param, verbose = FALSE)
write.csv(res, out_path)
