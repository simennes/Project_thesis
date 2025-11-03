# =========================
# Leave-One-Island-Out CV
# =========================

library(INLA)
library(dplyr)
library(data.table)
source("src/R/within_gp_example_func.R")

suppressWarnings({
  if (requireNamespace("INLA", quietly = TRUE)) {
    INLA::inla.setOption(inla.call = "inla")
  }
})

# ------------------------------
# Lightweight logger
# ------------------------------
log_msg <- function(..., level = "INFO", log_file = NULL) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  txt <- paste0("[", level, "] ", ts, " - ", paste(..., collapse = ""))
  message(txt)
  if (!is.null(log_file)) {
    cat(txt, "\n", file = log_file, append = TRUE)
  }
}

# ------------------------------
# Paths and PLINK executable
# ------------------------------
pheno_file <- "Data/AdultMorphology_20240201_fix.csv"

orig_geno_files <- paste0(
  "Data/",
  "combined_200k_70k_sparrow_genotype_data/",
  "combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05",
  c(".map", ".ped", ".fam", ".bim", ".bed")
)

get_plink_path <- function() {
  "C:/Users/Simen/OneDrive - NTNU/FYSMAT/INDMAT/25H/Prosjekt/PLINK/plink.exe"
}

# ------------------------------
# Model effects (include GRM)
# ------------------------------
grm_effect_strs <- sapply(
  effects(grm = TRUE),
  function(eff) make_effect_str(eff[[1]], eff[[2]])
)

# ------------------------------
# Genotyped IDs and global QC
# ------------------------------
genotyped_inds <- get_genotyped_inds(fam_file = orig_geno_files[3], sel = 1)

qc_filt <- list(
  genorate_ind = 0.05,
  genorate_snp = 0.1,
  maf = 0.01
)

qc_overall <- do_qc(
  fam_file  = orig_geno_files[3],
  ncores    = 8,
  mem       = 8 * 6000,
  qc_filt   = qc_filt,
  keep_inds = genotyped_inds,
  sys       = "",
  resp      = "overall"
)

genotyped_inds_qc <- get_genotyped_inds(fam_file = qc_overall[2], sel = 1)

# ------------------------------
# Phenotype & island selection
# ------------------------------
# Example: tarsus in Helgeland
response_colname <- "thr_tarsus"
response <- "tarsus"

isls <- c(20, 22, 23, 24, 26, 27, 28, 33, 34, 35, 38)
sys_name <- "helgeland"

# Prepare outputs and log file
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
log_file <- file.path("outputs", paste0("loo_island_cv_", response, "_", sys_name, ".log"))
log_msg(sprintf("Starting LOIO CV ??? response=%s, system=%s", response, sys_name), log_file = log_file)

pheno_data <- pheno_wrangle(
  filepath       = pheno_file,
  genotyped_inds = genotyped_inds_qc,
  islands        = isls,
  y_col_name     = response_colname,
  testing        = NULL
)

# ------------------------------
# QC restricted to analyzed inds
# ------------------------------
geno_files <- do_qc(
  fam_file  = qc_overall[2],
  ncores    = 8,
  mem       = 8 * 6000,
  qc_filt   = qc_filt,
  keep_inds = unique(pheno_data$ringnr),
  sys       = sys_name,
  resp      = response
)

# ------------------------------
# GRM (VanRaden) for analyzed inds
# ------------------------------
grm_files <- make_raw_grm(
  analysis_inds = unique(pheno_data$ringnr),
  bfile         = gsub(".{4}$", "", geno_files[2]),
  frq_file      = geno_files[5],
  genorate_ind  = qc_filt$genorate_ind,
  genorate_snp  = qc_filt$genorate_snp,
  ncores        = 8,
  mem           = 8 * 6000,
  maf           = qc_filt$maf,
  response      = response,
  geno_set      = paste0(sys_name, "_70K")
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
# Leave-One-Island-Out CV
# ------------------------------
all_inds <- unique(pheno_data$ringnr)
island_levels <- levels(pheno_data$island)
if (is.null(island_levels)) island_levels <- sort(unique(as.character(pheno_data$island)))

n_folds <- length(island_levels)
log_msg(sprintf("Prepared %d folds (islands): %s", n_folds, paste(island_levels, collapse = ", ")), log_file = log_file)

results <- vector("list", length(island_levels))
names(results) <- island_levels

for (ii in seq_along(island_levels)) {
  isl <- island_levels[ii]
  fold_start <- Sys.time()
  log_msg(sprintf("Fold %d/%d ??? island=%s: preparing train/test split", ii, n_folds, isl), log_file = log_file)
  
  # Test set = all observations from island 'isl'
  test_obs_idx <- which(as.character(pheno_data$island) == isl)
  test_inds <- unique(pheno_data$ringnr[test_obs_idx])
  
  # Train set = everyone else
  train_inds <- setdiff(all_inds, test_inds)
  log_msg(sprintf("Fold %d/%d ??? island=%s: n_obs_test=%d, n_inds_test=%d, n_inds_train=%d", 
                  ii, n_folds, isl, length(test_obs_idx), length(test_inds), length(train_inds)), 
          log_file = log_file)
  
  # Build prior from training variance
  var_train <- pheno_data %>%
    dplyr::filter(ringnr %in% train_inds) %>%
    dplyr::pull(y) %>%
    var()
  log_msg(sprintf("Fold %d/%d ??? island=%s: var_train=%.4f", ii, n_folds, isl, var_train), log_file = log_file)
  
  prior_fold <- make_prior(
    pc_prec_upper_var = var_train / 2,
    var_init          = var_train / 3,
    tau               = 0.05
  )
  
  # Fit GP model masking the test island
  fit <- NULL
  fit <- tryCatch({
    log_msg(sprintf("Fold %d/%d ??? island=%s: fitting model", ii, n_folds, isl), log_file = log_file)
    run_gp(
      pheno_data                 = pheno_data,
      train_inds                 = train_inds,
      test_inds                  = test_inds,
      inverse_relatedness_matrix = grm_obj$inv_grm,
      effects_vec                = grm_effect_strs,
      prior                      = prior_fold,
      ncores                     = 8
    )
  }, error = function(e) {
    log_msg(sprintf("Fold %d/%d ??? island=%s: ERROR during fit ??? %s", ii, n_folds, isl, conditionMessage(e)), level = "ERROR", log_file = log_file)
    NULL
  })
  
  if (is.null(fit)) {
    results[[ii]] <- data.frame(
      island     = isl,
      n_obs      = length(test_obs_idx),
      n_inds     = length(test_inds),
      pearson_r  = NA_real_,
      stringsAsFactors = FALSE
    )
    next
  }
  
  # Get predictions (posterior means of fitted values) for the test island rows
  yhat <- fit$model$summary.fitted.values$mean[test_obs_idx]
  yobs <- pheno_data$y[test_obs_idx]
  
  # Pearson correlation for this island
  r_pearson <- suppressWarnings(stats::cor(yhat, yobs, use = "complete.obs", method = "pearson"))
  fold_dur <- as.numeric(difftime(Sys.time(), fold_start, units = "secs"))
  log_msg(sprintf("Fold %d/%d ??? island=%s: pearson_r=%.4f, duration=%.1fs", ii, n_folds, isl, r_pearson, fold_dur), log_file = log_file)
  
  results[[ii]] <- data.frame(
    island     = isl,
    n_obs      = length(test_obs_idx),
    n_inds     = length(test_inds),
    pearson_r  = r_pearson,
    stringsAsFactors = FALSE
  )
}

res_df <- do.call(rbind, results)

# ------------------------------
# Save results
# ------------------------------
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
out_csv <- file.path("outputs", paste0("loo_island_cv_", response, "_", sys_name, ".csv"))
write.csv(res_df, out_csv, row.names = FALSE)
overall_r <- suppressWarnings(mean(res_df$pearson_r, na.rm = TRUE))
log_msg(sprintf("Completed LOIO CV ??? saved %d fold results to %s; overall mean Pearson r=%.4f", nrow(res_df), out_csv, overall_r), log_file = log_file)
