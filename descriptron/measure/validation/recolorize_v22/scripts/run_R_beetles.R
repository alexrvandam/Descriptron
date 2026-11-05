# recolorize 0.2.0 on its own example beetles. args: libpaths in_dir out_dir config
args <- commandArgs(trailingOnly = TRUE); .libPaths(c(strsplit(args[1], ":", fixed = TRUE)[[1]], .libPaths()))
suppressMessages({library(recolorize); library(png)})
in_dir <- args[2]; out_dir <- args[3]; config <- args[4]; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
rows <- list()
for (f in list.files(in_dir, pattern = "\\.png$", full.names = TRUE)) {
  n <- tools::file_path_sans_ext(basename(f)); set.seed(1)
  fit <- switch(config,
    default = recolorize2(f, plotting = FALSE),
    cut45   = recolorize2(f, cutoff = 45, plotting = FALSE),
    bins3   = recolorize2(f, bins = 3, plotting = FALSE))
  pa <- fit$pixel_assignments; writePNG(pa / 255, file.path(out_dir, paste0(n, "_fg_beetle_assign.png")))
  rows[[n]] <- data.frame(image = paste0(n, "_fg_beetle"), cluster = seq_len(nrow(fit$centers)),
                          R = round(fit$centers[, 1], 5), G = round(fit$centers[, 2], 5), B = round(fit$centers[, 3], 5), size = as.numeric(fit$sizes))
}
write.csv(do.call(rbind, rows), file.path(out_dir, "palettes.csv"), row.names = FALSE)
writeLines(c(paste("recolorize", packageVersion("recolorize")), R.version.string, paste("config", config)), file.path(out_dir, "settings.txt"))
