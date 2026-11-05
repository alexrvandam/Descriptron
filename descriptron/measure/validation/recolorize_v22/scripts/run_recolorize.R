# recolorize side.  Runs one configuration of recolorize 0.2.0 on every RGBA PNG in an input folder
# (alpha = Descriptron's _bin_ mask) and writes, per image, the pixel-assignment map and the palette.
# Usage: Rscript run_recolorize.R <rlib1:rlib2:...> <in_dir> <out_dir> <config> [file_list.txt]
# config:
#   default          recolorize2(img)                                  (histogram bins=2 in sRGB, recluster
#                                                                       cutoff=20 Lab, complete linkage, imposeColors)
#   b8_avg_T20       recolorize(img, "histogram", bins=8, sRGB) -> recluster(cutoff=20, hclust_method="average")
#   b8_complete_T20  recolorize(img, "histogram", bins=8, sRGB) -> recluster(cutoff=20, hclust_method="complete")
#   b8_avg_nfinal    as b8_avg_T20 but recluster(n_final = k per image from csv arg 6) - same number of colours
# Outputs in out_dir: <stem>_assign.png (8-bit grey, value = cluster index, 0 = background),
#                     palettes.csv (image, cluster, R, G, B in 0-1 sRGB, size), timing.csv, settings.txt
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(strsplit(args[1], ":", fixed = TRUE)[[1]], .libPaths()))
in_dir <- args[2]; out_dir <- args[3]; config <- args[4]
suppressMessages({library(recolorize); library(png)})
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
files <- sort(list.files(in_dir, pattern = "\\.png$", full.names = TRUE))
if (length(args) >= 5 && args[5] != "-") { keep <- readLines(args[5]); files <- files[basename(files) %in% keep] }
# b8_avg_nfinal: number of colours per image from a csv (columns image, k), arg 6
if (config == "b8_avg_nfinal") { kd <- read.csv(args[6]); kmap <- setNames(as.list(kd$k), kd$image) }
run_one <- function(f) {
  if (config == "default") {
    recolorize2(f, plotting = FALSE)
  } else if (config == "b8_avg_T20") {
    r1 <- recolorize(f, method = "histogram", bins = 8, color_space = "sRGB", plotting = FALSE)
    recluster(r1, cutoff = 20, hclust_method = "average", color_space = "Lab", ref_white = "D65",
              refit_method = "imposeColors", plot_hclust = FALSE, plot_final = FALSE)
  } else if (config == "b8_complete_T20") {
    r1 <- recolorize(f, method = "histogram", bins = 8, color_space = "sRGB", plotting = FALSE)
    recluster(r1, cutoff = 20, hclust_method = "complete", color_space = "Lab", ref_white = "D65",
              refit_method = "imposeColors", plot_hclust = FALSE, plot_final = FALSE)
  } else if (config == "b8_avg_nfinal") {
    r1 <- recolorize(f, method = "histogram", bins = 8, color_space = "sRGB", plotting = FALSE)
    kk <- kmap[[sub("\\.png$", "", basename(f))]]
    recluster(r1, n_final = min(kk, sum(r1$sizes > 0)), hclust_method = "average", color_space = "Lab", ref_white = "D65",
              refit_method = "imposeColors", plot_hclust = FALSE, plot_final = FALSE)
  } else stop("unknown config ", config)
}
writeLines(c(paste("recolorize", as.character(packageVersion("recolorize"))), paste("R", R.version.string),
             paste("config", config), paste("in_dir", in_dir), paste("n_images", length(files)),
             paste("started", format(Sys.time(), "%Y-%m-%dT%H:%M:%S"))), file.path(out_dir, "settings.txt"))
pal_file <- file.path(out_dir, "palettes.csv"); tim_file <- file.path(out_dir, "timing.csv")
done <- if (file.exists(tim_file)) read.csv(tim_file)$image else character(0)
if (!file.exists(pal_file)) write.csv(data.frame(image = character(0), cluster = integer(0), R = numeric(0), G = numeric(0),
                                                 B = numeric(0), size = numeric(0)), pal_file, row.names = FALSE)
if (!file.exists(tim_file)) write.csv(data.frame(image = character(0), seconds = numeric(0), k = integer(0)), tim_file, row.names = FALSE)
for (f in files) {
  stem <- sub("\\.png$", "", basename(f))
  if (stem %in% done) next
  t0 <- proc.time()[["elapsed"]]
  fit <- suppressMessages(run_one(f))
  secs <- proc.time()[["elapsed"]] - t0
  pa <- fit$pixel_assignments
  if (max(pa) > 255) stop("more than 255 clusters in ", stem)
  png::writePNG(pa / 255, file.path(out_dir, paste0(stem, "_assign.png")))
  ctr <- as.matrix(fit$centers)
  pal <- data.frame(image = stem, cluster = seq_len(nrow(ctr)), R = ctr[, 1], G = ctr[, 2], B = ctr[, 3], size = fit$sizes)
  write.table(pal, pal_file, sep = ",", append = TRUE, col.names = FALSE, row.names = FALSE)
  write.table(data.frame(image = stem, seconds = secs, k = sum(fit$sizes > 0)), tim_file, sep = ",", append = TRUE,
              col.names = FALSE, row.names = FALSE)
  cat(stem, round(secs, 1), "s, k =", sum(fit$sizes > 0), "\n")
}
cat("done", config, "\n")
