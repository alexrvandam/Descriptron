# Species colour-similarity trees from per-specimen palettes, the same way for every method:
#   specimen distance = earth mover's distance (emdist::emd, Euclidean ground distance in CIELAB = CIE76) between
#   pixel-weighted palettes (as in colordistance, Weller & Westneat 2019, the downstream recolorize exports to);
#   species distance  = mean EMD over all specimen pairs of the two species;
#   tree              = UPGMA (hclust average) -> ape::as.phylo.
# Compares every pair of methods: Mantel (vegan, Pearson, 9999 perm), cophenetic correlation of the UPGMA trees,
# Robinson-Foulds (phangorn, normalised; unrooted and rooted).  Noise floor: split-half within each method
# (alternate specimens of each species -> two independent species matrices).
# Optional: --colhom <colour.csv of the moth run>: Descriptron colour-homology features (species centroids of the
# standardised features, Euclidean) as one more matrix (Mantel only meaningful as context; it is a pattern trait).
# Usage: Rscript trees.R <rlibs> <compare_dir> <out_dir> <method1,method2,...> <pairs A:B;C:D> [colhom.csv]
a <- commandArgs(trailingOnly = TRUE)
.libPaths(c(strsplit(a[1], ":", fixed = TRUE)[[1]], .libPaths()))
suppressMessages({library(emdist); library(ape); library(phangorn); library(vegan)})
cmp <- a[2]; out <- a[3]; methods <- strsplit(a[4], ",")[[1]]
pairs <- lapply(strsplit(strsplit(a[5], ";")[[1]], ":"), identity)
colhom <- if (length(a) >= 6) a[6] else NA
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(1)

spec_emd <- function(pal) {
  ims <- unique(pal$image)
  P <- lapply(ims, function(i) { p <- pal[pal$image == i, ]; as.matrix(cbind(p$frac / sum(p$frac), p$L, p$a, p$b)) })
  n <- length(ims); D <- matrix(0, n, n, dimnames = list(ims, ims))
  for (i in 1:(n - 1)) for (j in (i + 1):n) D[i, j] <- D[j, i] <- emd(P[[i]], P[[j]])
  D
}
species_mat <- function(D, sp, keep = rep(TRUE, length(sp))) {
  S <- sort(unique(sp)); M <- matrix(0, length(S), length(S), dimnames = list(S, S))
  for (i in seq_along(S)) for (j in seq_along(S)) if (i != j) {
    ii <- which(sp == S[i] & keep); jj <- which(sp == S[j] & keep); M[i, j] <- mean(D[ii, jj, drop = FALSE]) }
  M
}
upgma <- function(M) as.phylo(hclust(as.dist(M), method = "average"))
cmp_pair <- function(MA, MB, tA, tB, label) {
  S <- intersect(rownames(MA), rownames(MB)); MA <- MA[S, S]; MB <- MB[S, S]
  tA <- keep.tip(tA, S); tB <- keep.tip(tB, S)
  mt <- mantel(as.dist(MA), as.dist(MB), method = "pearson", permutations = 9999)
  mts <- mantel(as.dist(MA), as.dist(MB), method = "spearman", permutations = 999)
  cA <- cophenetic(tA)[S, S]; cB <- cophenetic(tB)[S, S]
  data.frame(comparison = label, n_species = length(S), mantel_r = mt$statistic, mantel_P = mt$signif,
             mantel_spearman = mts$statistic, cophenetic_r = cor(as.dist(cA), as.dist(cB)),
             RF_unrooted_norm = RF.dist(unroot(tA), unroot(tB), normalize = TRUE),
             RF_rooted_norm = RF.dist(tA, tB, normalize = TRUE, rooted = TRUE),
             RF_unrooted_raw = RF.dist(unroot(tA), unroot(tB)))
}
M <- list(); Tr <- list(); half <- list(); rows <- list()
for (m in methods) {
  pal <- read.csv(file.path(cmp, paste0("palettes_", m, ".csv")))
  D <- spec_emd(pal); sp <- sub("__.*", "", rownames(D))
  write.csv(D, file.path(out, paste0("specimen_emd_", m, ".csv")))
  M[[m]] <- species_mat(D, sp); Tr[[m]] <- upgma(M[[m]])
  write.csv(M[[m]], file.path(out, paste0("species_dist_", m, ".csv"))); write.tree(Tr[[m]], file.path(out, paste0("upgma_", m, ".nwk")))
  idx <- ave(seq_along(sp), sp, FUN = seq_along); hA <- idx %% 2 == 1
  MA <- species_mat(D, sp, hA); MB <- species_mat(D, sp, !hA)
  half[[m]] <- list(A = MA, B = MB)
  if (all(is.finite(MA)) && all(is.finite(MB))) rows[[length(rows) + 1]] <- cmp_pair(MA, MB, upgma(MA), upgma(MB), paste0("split-half within ", m))
  cat(m, ":", nrow(D), "specimens,", nrow(M[[m]]), "species\n")
}
for (p in pairs) {
  if (all(p %in% names(M))) {
    rows[[length(rows) + 1]] <- cmp_pair(M[[p[1]]], M[[p[2]]], Tr[[p[1]]], Tr[[p[2]]], paste(p[1], "vs", p[2]))
    # cross-method split half: half A of method 1 vs half B of method 2 (sampling noise + method difference)
    hA <- half[[p[1]]]$A; hB <- half[[p[2]]]$B
    if (all(is.finite(hA)) && all(is.finite(hB))) rows[[length(rows) + 1]] <- cmp_pair(hA, hB, upgma(hA), upgma(hB), paste(p[1], "halfA vs", p[2], "halfB"))
  }
}
if (!is.na(colhom) && file.exists(colhom)) {
  d <- read.csv(colhom, check.names = FALSE); X <- scale(as.matrix(d[, -(1:2)])); X[is.na(X)] <- 0
  cen <- apply(X, 2, function(v) tapply(v, d$species, mean)); MC <- as.matrix(dist(cen))
  write.csv(MC, file.path(out, "species_dist_colour_homology.csv"))
  for (m in methods) rows[[length(rows) + 1]] <- cmp_pair(MC, M[[m]], upgma(MC), Tr[[m]], paste("colour_homology vs", m))
}
res <- do.call(rbind, rows); rownames(res) <- NULL
write.csv(res, file.path(out, "tree_comparisons.csv"), row.names = FALSE)
print(res, digits = 3)
