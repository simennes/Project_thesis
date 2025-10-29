# =========================
# Preprocess for INLA LOIO GBLUP (runs locally with PLINK)
# =========================

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
})

# Utilities and GP helpers
source("src/R/within_gp_example_func.R")

# ------------------------------
# Config (adjust as needed)
# ------------------------------
pheno_file <- "Data/AdultMorphology_20240201_fix.csv"

# Original PLINK files (MAP/PED/FAM/BIM/BED) base path
orig_geno_files <- paste0(
  "Data/",
  "combined_200k_70k_sparrow_genotype_data/",
  "combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05",
  c(".map", ".ped", ".fam", ".bim", ".bed")
)

# Define how to find PLINK on this machine
get_plink_path <- function() {
  # Prefer environment variable; fallback to 'plink' on PATH
  p <- Sys.getenv("PLINK", unset = NA)
  if (!is.na(p) && nzchar(p)) return(p)
  return("plink")
}

# Phenotype target and islands to include in analysis universe
response_colname <- "thr_tarsus"
response_label   <- "tarsus"
isls <- c(20,22,23,24,26,27,28,33,34,35,38)

# QC thresholds for PLINK
qc_filt <- list(
  genorate_ind = 0.05,
  genorate_snp = 0.10,
  maf          = 0.01
)

# Output directory for artifacts consumed by cluster LOIO
prep_out_dir <- file.path("outputs", "prep_inla_gblup")
dir.create(prep_out_dir, showWarnings = FALSE, recursive = TRUE)

log_msg <- function(...) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  message(sprintf("[%s] %s", ts, paste(..., collapse = "")))
}

# ------------------------------
# 1) Read genotyped IDs and run global QC (PLINK)
# ------------------------------
log_msg("Reading genotyped IDs from FAM and running global QC …")
all_genotyped <- get_genotyped_inds(fam_file = orig_geno_files[3], sel = 1)

qc_overall <- do_qc(
  fam_file  = orig_geno_files[3],
  ncores    = 8,
  mem       = 8 * 6000,
  qc_filt   = qc_filt,
  keep_inds = all_genotyped,
  sys       = "overall",
  resp      = "global"
)

genotyped_qc <- get_genotyped_inds(fam_file = qc_overall[2], sel = 1)

# ------------------------------
# 2) Phenotype wrangling and island selection
# ------------------------------
log_msg("Wrangling phenotype and selecting target islands …")
pheno_data <- pheno_wrangle(
  filepath       = pheno_file,
  genotyped_inds = genotyped_qc,
  islands        = isls,
  y_col_name     = response_colname,
  testing        = NULL
)

# ------------------------------
# 3) QC restricted to analyzed individuals (PLINK)
# ------------------------------
log_msg("Running PLINK QC restricted to analyzed individuals …")
geno_files <- do_qc(
  fam_file  = qc_overall[2],
  ncores    = 8,
  mem       = 8 * 6000,
  qc_filt   = qc_filt,
  keep_inds = unique(pheno_data$ringnr),
  sys       = "helgeland",
  resp      = response_label
)

# ------------------------------
# 4) Build GRM for analyzed set (PLINK) and compute inverse
# ------------------------------
log_msg("Building GRM and computing inverse precision matrix …")
grm_files <- make_raw_grm(
  analysis_inds = unique(pheno_data$ringnr),
  bfile         = gsub(".{4}$", "", geno_files[2]),
  frq_file      = geno_files[5],
  genorate_ind  = qc_filt$genorate_ind,
  genorate_snp  = qc_filt$genorate_snp,
  ncores        = 8,
  mem           = 8 * 6000,
  maf           = qc_filt$maf,
  response      = response_label,
  geno_set      = paste0("helgeland_","70K")
)

grm_obj <- compute_grm_obj(
  frq_file   = geno_files[5],
  rel_file   = grm_files[2],
  id_file    = grm_files[1],
  bim_file   = grm_files[3],
  pheno_data = pheno_data,
  id_col     = 1
)

# ------------------------------
# 5) Save artifacts for cluster LOIO
# ------------------------------
log_msg("Saving preprocessed artifacts for cluster execution …")
# Save phenotype data (compact: only columns needed for run_gp and LOIO)
pheno_keep <- c("ringnr","y","sex","year","month","island","hatch_year","first_island","age","day_session","id1","id2")
pheno_save <- pheno_data[, intersect(pheno_keep, colnames(pheno_data)), drop = FALSE]

saveRDS(pheno_save, file.path(prep_out_dir, "pheno_data.Rds"))
saveRDS(grm_obj$inv_grm, file.path(prep_out_dir, "inv_grm.Rds"))
# Also save ordering of IDs for safety
if (!is.null(rownames(grm_obj$inv_grm))) {
  write.csv(data.frame(id_order = rownames(grm_obj$inv_grm)), file.path(prep_out_dir, "inv_grm_id_order.csv"), row.names = FALSE)
}

meta <- list(
  response_colname = response_colname,
  response_label   = response_label,
  islands          = isls,
  prep_timestamp   = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)
jsonlite::write_json(meta, file.path(prep_out_dir, "meta.json"), auto_unbox = TRUE, pretty = TRUE)

log_msg(sprintf("Done. Artifacts written to: %s", prep_out_dir))
