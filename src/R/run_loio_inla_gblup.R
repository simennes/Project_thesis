# =========================
# Run INLA LOIO GBLUP (cluster-safe; no PLINK required)
# =========================

suppressPackageStartupMessages({
  library(INLA)
  library(dplyr)
  library(jsonlite)
})

# Utilities and GP helpers
source("src/R/within_gp_example_func.R")

# ------------------------------
# Config (adjust as needed)
# ------------------------------
# Directory containing artifacts produced by preprocess_gblup.R
prep_in_dir <- file.path("outputs", "prep_inla_gblup")

# Output results file (JSON) similar to nested_cv_loio_nograph_results.json
out_dir <- file.path("outputs", "nested_cv")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
results_json <- file.path(out_dir, "nested_cv_loio_inla_gblup_results.json")

# Optional: log file
log_file <- file.path(out_dir, "loio_inla_gblup.log")

log_msg <- function(..., level = "INFO", log_file = NULL) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  txt <- paste0("[", level, "] ", ts, " - ", paste(..., collapse = ""))
  message(txt)
  if (!is.null(log_file)) cat(txt, "\n", file = log_file, append = TRUE)
}

# ------------------------------
# 1) Load artifacts
# ------------------------------
if (!file.exists(file.path(prep_in_dir, "pheno_data.Rds"))) stop("Missing pheno_data.Rds in ", prep_in_dir)
if (!file.exists(file.path(prep_in_dir, "inv_grm.Rds"))) stop("Missing inv_grm.Rds in ", prep_in_dir)

pheno_data <- readRDS(file.path(prep_in_dir, "pheno_data.Rds"))
inv_grm     <- readRDS(file.path(prep_in_dir, "inv_grm.Rds"))
meta        <- tryCatch(jsonlite::read_json(file.path(prep_in_dir, "meta.json")), error = function(e) NULL)

# ------------------------------
# 2) Effects (fixed + random + GRM)
# ------------------------------
# Build INLA effect strings including GRM term using helper utilities
# effects(grm=TRUE) should return list of effect specs; make_effect_str converts to INLA formula string components
# (see within_gp_example_func.R)
grm_effect_strs <- sapply(effects(grm = TRUE), function(eff) make_effect_str(eff[[1]], eff[[2]]))

# ------------------------------
# 3) LOIO CV
# ------------------------------
all_inds <- unique(pheno_data$ringnr)
island_levels <- levels(pheno_data$island)
if (is.null(island_levels)) island_levels <- sort(unique(as.character(pheno_data$island)))

n_folds <- length(island_levels)
log_msg(sprintf("Prepared %d folds (islands): %s", n_folds, paste(island_levels, collapse = ", ")), log_file = log_file)

outer_r <- numeric(0)
per_fold_rows <- list()

for (ii in seq_along(island_levels)) {
  isl <- island_levels[ii]
  fold_start <- Sys.time()
  log_msg(sprintf("Fold %d/%d :: island=%s :: preparing train/test split", ii, n_folds, isl), log_file = log_file)

  # Test = all obs from island
  test_obs_idx <- which(as.character(pheno_data$island) == isl)
  test_inds <- unique(pheno_data$ringnr[test_obs_idx])
  # Train = all others
  train_inds <- setdiff(all_inds, test_inds)

  # Prior from training variance
  var_train <- pheno_data %>% dplyr::filter(ringnr %in% train_inds) %>% dplyr::pull(y) %>% var()
  prior_fold <- make_prior(pc_prec_upper_var = var_train / 2, var_init = var_train / 3, tau = 0.05)

  # Fit GP with masked test y
  fit <- NULL
  fit <- tryCatch({
    log_msg(sprintf("Fold %d/%d :: island=%s :: fitting model", ii, n_folds, isl), log_file = log_file)
    run_gp(
      pheno_data                 = pheno_data,
      train_inds                 = train_inds,
      test_inds                  = test_inds,
      inverse_relatedness_matrix = inv_grm,
      effects_vec                = grm_effect_strs,
      prior                      = prior_fold,
      ncores                     = 8
    )
  }, error = function(e) {
    log_msg(sprintf("Fold %d/%d :: island=%s :: ERROR :: %s", ii, n_folds, isl, conditionMessage(e)), level = "ERROR", log_file = log_file)
    NULL
  })

  if (is.null(fit)) {
    outer_r <- c(outer_r, NA_real_)
    per_fold_rows[[ii]] <- list(fold = ii, island = as.character(isl), pearson_r = NA_real_)
    next
  }

  # Predict and compute Pearson r on held-out island
  yhat <- fit$model$summary.fitted.values$mean[test_obs_idx]
  yobs <- pheno_data$y[test_obs_idx]
  rr <- suppressWarnings(stats::cor(yhat, yobs, use = "complete.obs", method = "pearson"))
  outer_r <- c(outer_r, rr)
  fold_dur <- as.numeric(difftime(Sys.time(), fold_start, units = "secs"))
  log_msg(sprintf("Fold %d/%d :: island=%s :: pearson_r=%.4f :: %.1fs", ii, n_folds, isl, rr, fold_dur), log_file = log_file)
  per_fold_rows[[ii]] <- list(fold = ii, island = as.character(isl), pearson_r = rr)
}

# ------------------------------
# 4) Save JSON results (nested_cv style)
# ------------------------------
summary_obj <- list(
  mode = "transductive",
  cv_strategy = "leave_island_out",
  outer_test_corr = as.numeric(outer_r),
  outer_test_corr_mean = if (all(is.na(outer_r))) NULL else mean(outer_r, na.rm = TRUE),
  outer_test_corr_std  = if (all(is.na(outer_r))) NULL else stats::sd(outer_r, na.rm = TRUE),
  inner_splits = 0L,
  outer_splits = length(island_levels),
  best_params_per_fold = list() # none for INLA run; keep key for compatibility
)

jsonlite::write_json(summary_obj, results_json, auto_unbox = TRUE, pretty = TRUE)
log_msg(sprintf("Wrote results to %s", results_json), log_file = log_file)
