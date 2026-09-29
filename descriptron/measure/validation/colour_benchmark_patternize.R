#!/usr/bin/env Rscript
# patternize arm of the colour-pattern benchmark.
#
# Workflow = the package's documented landmark + k-means route
# (patLanK -> choose the k-means cluster of interest -> maskOutline -> patPCA),
# as in StevenVB12/patternize-examples/examples_manuscript.R (Fig. 4 k-means + PCA
# block) and the patLanK help page. Settings: patLanK defaults (k = 3, res = 300,
# transformRef = 'meanshape', transformType = 'tps', resampleFactor = NULL i.e. full
# resolution), with crop = TRUE and adjustCoords = TRUE as in the documented examples,
# and cropOffset = c(5,5,5,5): the help page says to set it when "the landmarks do not
# surround the entire color pattern" -- here the wing margin bulges beyond landmarks
# 1-11 (with the default 0 the costal and anal margins were clipped; see README).
#
# Usage:
#   Rscript colour_benchmark_patternize.R --lib <Rlib> --inputs <prepare outdir> \
#       --outdir <dir> [--k 3] [--res 300] [--resample 0 (=NULL)] [--seed 1] [--crop_offset 5,5,5,5] [--reuse]
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 0) return(default)
  args[i + 1]
}
lib <- getarg("--lib"); inputs <- getarg("--inputs"); outdir <- getarg("--outdir")
if (is.null(inputs) || is.null(outdir)) stop("need --inputs and --outdir")
if (!is.null(lib)) .libPaths(c(lib, .libPaths()))
k <- as.integer(getarg("--k", 3)); res <- as.integer(getarg("--res", 300))
resample <- as.integer(getarg("--resample", 0)); resampleArg <- if (resample > 0) resample else NULL
reuse <- "--reuse" %in% args
cropOffset <- as.numeric(strsplit(getarg("--crop_offset", "5,5,5,5"), ",")[[1]]); seed <- as.integer(getarg("--seed", 1))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(outdir, "qc"), showWarnings = FALSE)

suppressPackageStartupMessages({library(patternize); library(raster)})
cat("patternize", as.character(packageVersion("patternize")),
    "Morpho", as.character(packageVersion("Morpho")),
    "raster", as.character(packageVersion("raster")), "\n")

spec <- read.csv(file.path(inputs, "specimens.csv"), stringsAsFactors = FALSE)
IDlist <- spec$id
cartoonID <- spec$id[spec$patternize_cartoon == 1]

t0 <- Sys.time()
landmarkList <- makeList(IDlist, "landmark", file.path(inputs, "landmarks"), "_landmarks.txt")
imageList <- makeList(IDlist, "image", file.path(inputs, "images"), ".png")

# patLanK prints the k-means start centres of the first image; keep the log so the
# cluster order can be read back (the same centres seed every later image).
logf <- file.path(outdir, "patLanK_log.txt")
rda <- file.path(outdir, "rasterList_lanK.rda")
if (reuse && file.exists(rda) && file.exists(logf)) {
  load(rda); t_align <- NA
  cat("reusing", rda, "\n")
} else {
  set.seed(seed)
  zz <- file(logf, open = "wt"); sink(zz, split = TRUE)
  rasterList_lanK <- patLanK(imageList, landmarkList, k = k, resampleFactor = resampleArg,
                             crop = TRUE, cropOffset = cropOffset, res = res, transformRef = "meanshape",
                             adjustCoords = TRUE, plot = FALSE)
  sink(); close(zz)
  t_align <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  save(rasterList_lanK, file = rda)
}

missing <- setdiff(IDlist, names(rasterList_lanK))
if (length(missing)) cat("WARNING: k-means failed/skipped for", missing, "\n")

# read start centres (k rows of R G B) from the log
lg <- readLines(logf)
st <- grep("start centers of first image", lg)
cen <- t(sapply(lg[(st + 2):(st + 1 + k)], function(l) {
  v <- as.numeric(strsplit(trimws(l), "\\s+")[[1]]); v[(length(v) - 2):length(v)]
}))
rownames(cen) <- NULL
bright <- rowMeans(cen)
darkest <- which.min(bright)
cat("start centres (RGB):\n"); print(cen)
cat("cluster of interest = darkest centre =", darkest, "\n")
write.csv(data.frame(cluster = 1:k, R = cen[, 1], G = cen[, 2], B = cen[, 3],
                     mean = bright, chosen = (1:k) == darkest),
          file.path(outdir, "kmeans_start_centres.csv"), row.names = FALSE)

# outline of the cartoon specimen (whole_wing polygon, image pixel coords, y down)
outline <- read.table(file.path(inputs, "outlines", paste0(cartoonID, "_outline.txt")))

mask_one <- function(r) maskOutline(r, outline, refShape = "mean", landList = landmarkList,
                                    adjustCoords = TRUE, cartoonID = cartoonID,
                                    IDlist = IDlist, imageList = imageList)
t1 <- Sys.time()
sel <- list(); allk <- list()
for (id in IDlist) {
  sel[[id]] <- mask_one(rasterList_lanK[[id]][[darkest]])
  allk[[id]] <- lapply(1:k, function(j) mask_one(rasterList_lanK[[id]][[j]]))
}
t_mask <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

# pixel table exactly as patPCA builds it (NA -> 0, one column per sample)
to_df <- function(rl) {
  m <- sapply(IDlist, function(id) { r <- rl[[id]]; r[is.na(r)] <- 0; raster::values(r) })
  t(m)
}
Xsel <- to_df(sel)
Xall <- do.call(cbind, lapply(1:k, function(j) to_df(lapply(allk, function(x) x[[j]]))))
write.csv(data.frame(id = IDlist, Xsel, check.names = FALSE),
          file.path(outdir, "patternize_features.csv"), row.names = FALSE)
write.csv(data.frame(id = IDlist, Xall, check.names = FALSE),
          file.path(outdir, "patternize_allk_features.csv"), row.names = FALSE)

# the package's own PCA on the same rasters (check that it matches ours)
popList <- lapply(split(spec$id, spec$species), identity)
colList <- rainbow(length(popList))
pc <- patPCA(sel, popList, colList, plot = FALSE)
write.csv(data.frame(id = rownames(pc$x), pc$x[, 1:10]),
          file.path(outdir, "patternize_patPCA_scores.csv"), row.names = FALSE)

# QC: one aligned + masked raster, the heat map of the chosen cluster, and the
# per-cluster heat maps (summed over all wings)
png(file.path(outdir, "qc", "patternize_aligned_one_wing.png"), 1500, 520)
par(mfrow = c(1, 3))
plotRGB(imageList[[IDlist[1]]], main = IDlist[1])
plot(rasterList_lanK[[IDlist[1]]][[darkest]], main = "aligned cluster (before mask)", col = "black", legend = FALSE)
plot(sel[[IDlist[1]]], main = "aligned + masked (to mean shape)")
dev.off()
summed <- sumRaster(sel, IDlist, type = "RGB")
png(file.path(outdir, "qc", "patternize_heatmap_chosen_cluster.png"), 900, 700)
plot(summed / length(IDlist), main = "frequency of darkest k-means cluster (masked, mean shape)")
dev.off()
# the outline mask itself, in mean-shape space, over the first wing's aligned cluster
ones <- rasterList_lanK[[IDlist[1]]][[darkest]]; ones[] <- 1
png(file.path(outdir, "qc", "patternize_outline_mask.png"), 1200, 520)
par(mfrow = c(1, 2))
plot(mask_one(ones), main = "maskOutline region (1 = kept)")
plot(sumRaster(lapply(rasterList_lanK, function(x) x[[3]]), IDlist, type = "RGB") / length(IDlist),
     main = "cluster 3 frequency, unmasked (shows wing extent)")
dev.off()
summedK <- sumRaster(rasterList_lanK, IDlist, type = "k")
png(file.path(outdir, "qc", "patternize_heatmap_all_clusters.png"), 600 * k, 520)
par(mfrow = c(1, k))
for (j in 1:k) plot(summedK[[j]] / length(IDlist), main = paste("cluster", j, "(unmasked)"))
dev.off()
# stack of all wings for mirrored vs not: mean raster for mirrored wings only
mir <- spec$id[spec$mirror_vs_majority == 1]; nonm <- spec$id[spec$mirror_vs_majority == 0]
png(file.path(outdir, "qc", "patternize_mirrored_vs_not.png"), 1200, 520)
par(mfrow = c(1, 2))
plot(sumRaster(sel[nonm], nonm, type = "RGB") / length(nonm), main = "non-mirrored wings", zlim = c(0, 1))
plot(sumRaster(sel[mir], mir, type = "RGB") / length(mir), main = "mirrored wings", zlim = c(0, 1))
dev.off()

tt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
writeLines(c(sprintf("align_kmeans_seconds=%.1f", t_align), sprintf("mask_seconds=%.1f", t_mask),
             sprintf("total_seconds=%.1f", tt), sprintf("n_wings_out=%d", length(rasterList_lanK)),
             sprintf("k=%d", k), sprintf("cropOffset=%s", paste(cropOffset, collapse = ",")), sprintf("res=%d", res), sprintf("resampleFactor=%s", ifelse(resample > 0, resample, "NULL")),
             sprintf("chosen_cluster=%d", darkest), sprintf("cartoonID=%s", cartoonID),
             sprintf("n_pixels=%d", ncol(Xsel))),
           file.path(outdir, "patternize_run_info.txt"))
cat("done in", round(tt), "s\n")
