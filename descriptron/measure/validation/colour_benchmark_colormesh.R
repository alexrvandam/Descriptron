#!/usr/bin/env Rscript
# Colormesh arm of the colour-pattern benchmark.
#
# Workflow = the Colormesh (V2.0) README: landmarks placed externally (sec. 2.2.1.2),
# perimeter map + sliders (make.sliders, sec. 2.2.2.1), tps.unwarp to the consensus
# (sec. 2.2.2.2), tri.surf Delaunay sampling template with num.passes = 3 (sec. 2.4.1),
# rgb.measure with px.radius = 2, linearize.color.space = FALSE (sec. 2.4.2.2),
# make.colormesh.dataset(use.perimeter.data = TRUE) (sec. 3).
# No colour calibration (rgb.calibrate): the slides carry no colour standard.
#
# Usage:
#   Rscript colour_benchmark_colormesh.R --lib <Rlib> --inputs <prepare outdir> \
#       --outdir <dir> [--passes 3] [--radius 2] [--flip auto|TRUE|FALSE]
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 0) return(default)
  args[i + 1]
}
lib <- getarg("--lib"); inputs <- getarg("--inputs"); outdir <- getarg("--outdir")
if (is.null(inputs) || is.null(outdir)) stop("need --inputs and --outdir")
if (!is.null(lib)) .libPaths(c(lib, .libPaths()))
passes <- as.integer(getarg("--passes", 3)); radius <- as.integer(getarg("--radius", 2))
flip <- getarg("--flip", "TRUE")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(outdir, "qc"), showWarnings = FALSE)
unw <- file.path(outdir, "unwarped")
dir.create(unw, showWarnings = FALSE)

suppressPackageStartupMessages({library(Colormesh); library(imager); library(jsonlite)})
cat("Colormesh", as.character(packageVersion("Colormesh")),
    "geomorph", as.character(packageVersion("geomorph")), "\n")

spec <- read.csv(file.path(inputs, "specimens.csv"), stringsAsFactors = FALSE)
ids <- spec$id
pm <- fromJSON(file.path(inputs, "colormesh_perimeter.json"))
perimeter.map <- pm$perimeter_map
lmdf <- read.csv(file.path(inputs, "colormesh_landmarks.csv"), stringsAsFactors = FALSE)
p <- max(lmdf$point)
A <- array(NA_real_, dim = c(p, 2, length(ids)), dimnames = list(NULL, c("x", "y"), ids))
for (id in ids) {
  s <- lmdf[lmdf$id == id, ]
  s <- s[order(s$point), ]
  A[, 1, id] <- s$x
  A[, 2, id] <- s$y_tps          # TPS convention: origin bottom-left, as Colormesh expects
}
stopifnot(!anyNA(A))
sliders <- make.sliders(perimeter.map, main.lms = pm$main_landmarks)

# tps.unwarp loops over every image file in imagedir: the prepare step writes exactly
# the benchmark wings there.
imgdir <- paste0(normalizePath(file.path(inputs, "images")), "/")
stopifnot(setequal(tools::file_path_sans_ext(list.files(imgdir)), ids))
t0 <- Sys.time()
unwarped <- tps.unwarp(imagedir = imgdir, landmarks = A, image.names = ids,
                       sliders = sliders, write.dir = unw)
t_unwarp <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
saveRDS(unwarped, file.path(outdir, "tps_unwarp_result.rds"))

# sampling template; check alignment for both flip settings (README 2.4.1.1)
test_img <- load.image(file.path(unw, unwarped$unwarped.names[1]))
for (fl in c(FALSE, TRUE)) {
  png(file.path(outdir, "qc", sprintf("colormesh_template_flip_%s.png", fl)), 900, 900)
  tri.surf(unwarped$target, perimeter.map, num.passes = passes,
           corresponding.image = test_img, flip.delaunay = fl)
  dev.off()
}
flip_val <- as.logical(flip)
png(file.path(outdir, "qc", "colormesh_template_used.png"), 900, 900)
template <- tri.surf(unwarped$target, perimeter.map, num.passes = passes,
                     corresponding.image = test_img, flip.delaunay = flip_val)
dev.off()
cat("sampling points: interior", nrow(template$interior), "perimeter", nrow(template$perimeter), "\n")

t1 <- Sys.time()
uncalib <- rgb.measure(imagedir = paste0(normalizePath(unw), "/"), image.names = unwarped$unwarped.names,
                       delaunay.map = template, px.radius = radius, linearize.color.space = FALSE)
t_measure <- as.numeric(difftime(Sys.time(), t1, units = "secs"))
saveRDS(uncalib, file.path(outdir, "rgb_measure_result.rds"))

sf <- data.frame(image = ids, species = spec$species)
final.df <- make.colormesh.dataset(df = uncalib, specimen.factors = sf, use.perimeter.data = TRUE)
write.csv(final.df, file.path(outdir, "colormesh_dataset_uncalib.csv"), row.names = FALSE)

# feature table: RGB of every sampled point (interior + perimeter), same order as ids
nm <- tools::file_path_sans_ext(sub("_unwarped$", "", tools::file_path_sans_ext(dimnames(uncalib$sampled.color)[[3]])))
stopifnot(identical(nm, ids))
flat <- function(a) t(apply(a, 3, function(m) as.vector(m)))   # n x (points*3), R block then G then B
Xi <- flat(uncalib$sampled.color); Xp <- flat(uncalib$sampled.perimeter)
colnames(Xi) <- paste0("int_", rep(c("R", "G", "B"), each = dim(uncalib$sampled.color)[1]), "_", seq_len(dim(uncalib$sampled.color)[1]))
colnames(Xp) <- paste0("per_", rep(c("R", "G", "B"), each = dim(uncalib$sampled.perimeter)[1]), "_", seq_len(dim(uncalib$sampled.perimeter)[1]))
write.csv(data.frame(id = ids, Xi, Xp, check.names = FALSE), file.path(outdir, "colormesh_features.csv"), row.names = FALSE)
write.csv(data.frame(id = ids, Xi, check.names = FALSE), file.path(outdir, "colormesh_interior_features.csv"), row.names = FALSE)

# QC overlays: sampling points on two unwarped wings (one mirrored in the originals)
for (id in c(ids[1], spec$id[spec$mirror_vs_majority == 1][1])) {
  im <- load.image(file.path(unw, paste0(id, "_unwarped.png")))
  png(file.path(outdir, "qc", paste0("colormesh_points_", id, ".png")), 900, 900)
  plot(im, main = id)
  points(template$interior, col = "red", pch = 20, cex = 0.6)
  points(template$perimeter, col = "yellow", pch = 20, cex = 0.8)
  dev.off()
}
png(file.path(outdir, "qc", "colormesh_sampled_colour_wing1.png"), 900, 700)
plot(uncalib, individual = 1, style = "comparison")
dev.off()

tt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
writeLines(c(sprintf("unwarp_seconds=%.1f", t_unwarp), sprintf("measure_seconds=%.1f", t_measure),
             sprintf("total_seconds=%.1f", tt), sprintf("num_passes=%d", passes),
             sprintf("px_radius=%d", radius), sprintf("flip_delaunay=%s", flip_val),
             sprintf("n_landmarks_total=%d", p), sprintf("n_semilandmarks=%d", pm$n_semilandmarks),
             sprintf("n_interior_points=%d", nrow(template$interior)),
             sprintf("n_perimeter_points=%d", nrow(template$perimeter))),
           file.path(outdir, "colormesh_run_info.txt"))
cat("done in", round(tt), "s\n")
