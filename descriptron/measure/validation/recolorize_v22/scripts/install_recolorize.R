# Install recolorize (and emdist, clue) into a throwaway library given as arg 1.
# Extra read-only library paths (already-installed deps such as imager) are args 2..n.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; extra <- args[-1]
.libPaths(c(lib, extra, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 1)
need <- c("recolorize", "emdist", "clue")
for (p in need) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib, dependencies = c("Depends", "Imports", "LinkingTo"))
for (p in need) cat(p, as.character(packageVersion(p)), "\n")
cat("installed into", lib, ":\n"); print(list.files(lib))
