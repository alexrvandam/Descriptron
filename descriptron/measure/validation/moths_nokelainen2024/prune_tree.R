# Noctuidae + Erebidae clade of the authors' ultrametric tree (Nokelainen et al. 2024 Nat Commun source data),
# tip names mapped to the wing-image species names used for the GBIF images.
args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths())); suppressMessages(library(ape))
tr <- read.tree(args[2])
m <- getMRCA(tr, c("Autographa_californica", "Grammia_nevadensis"))
cl <- extract.clade(tr, m)
map <- c(Orthosia_hibisci_brucei = "Orthosia_hibisci", Calliteara_taiwana = "Calliteara_pudibunda",
         Herminia_tarsicrinalis = "Herminia_grisealis", Euplagia_quadripunctata = "Euplagia_quadripunctaria",
         Schranckia_taenialis = "Schrankia_taenialis", Nudaria_mundane = "Nudaria_mundana",
         Arctia_villica_britannica = "Arctia_villica")
i <- cl$tip.label %in% names(map); cl$tip.label[i] <- map[cl$tip.label[i]]
write.tree(cl, args[3]); cat("clade tips:", Ntip(cl), " ultrametric:", is.ultrametric(cl), "\n")
