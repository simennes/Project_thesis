# support_genomics_plink.R
# PLINK-driven F-hat (inbreeding) + LMM adjusted phenotypes

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lme4)
})

# ------------------------------ utils ----------------------------------------
.center_scale <- function(x, center = TRUE, scale = FALSE) {
  if (!center && !scale) return(x)
  m <- if (center) mean(x, na.rm = TRUE) else 0
  s <- if (scale) sd(x, na.rm = TRUE) else 1
  if (is.na(s) || s == 0) s <- 1
  if (center && !scale) return(x - m)
  if (!center && scale)  return(x / s)
  (x - m) / s
}

.plink_run <- function(plink_bin, args, must_exist = NULL) {
  # Use system2 with vectorized args for safe quoting across OSes.
  res <- system2(plink_bin, args, stdout = TRUE, stderr = TRUE)
  code <- attr(res, "status")
  if (!is.null(code) && code != 0) {
    stop("PLINK failed (exit code ", code, "). Output:\n", paste(res, collapse = "\n"))
  }
  if (!is.null(must_exist)) {
    if (!file.exists(must_exist)) {
      stop("PLINK expected output not found: ", must_exist)
    }
  }
  invisible(TRUE)
}

# --------------------- QC + subset (mirrors your GRM flow) -------------------
# 1) Drop flagged samples (HIGHHET/MISSEX in IID), keep last IID per FID
# 2) Overall QC: --maf/--geno/--mind, --make-bed --freq
# 3) Trait/system subset QC: keep phenotyped ringnr, re-run the same QC, --freq
#
# Returns list(bfile_qc_sub, frq_file, keep_file)

plink_qc_and_subset <- function(plink_bin,
                                bfile_raw,
                                ph_df,
                                ringnr_col = "ringnr",
                                outdir_overall = "Data/qc_overall_",
                                outdir_subset = "Data/qc_subset",
                                maf_min = 0.01,
                                geno_max = 0.10,
                                mind_max = 0.05,
                                chrset = 32,
                                drop_iid_regex = "HIGHHET|MISSEX") {
  # Load FAM to build keep lists
  fam <- fread(paste0(bfile_raw, ".fam"), header = FALSE)
  setnames(fam, c("V1","V2"), c("FID","IID"))
  fam[, FID := as.character(FID)]
  fam[, IID := as.character(IID)]
  
  # 1) Drop flagged samples and keep the last IID per FID
  fam_keep <- fam[!grepl(drop_iid_regex, IID)]
  fam_keep <- fam_keep[!duplicated(fam_keep$FID, fromLast = TRUE)]  # keep last IID per FID
  
  dir.create(outdir_overall, recursive = TRUE, showWarnings = FALSE)
  keep_overall <- file.path(outdir_overall, "keep.txt")
  fwrite(fam_keep[, .(FID, IID)], keep_overall, col.names = FALSE, sep = "\t")
  
  # 2) Overall QC
  bfile_qc_all <- file.path(outdir_overall, "qc")
  args1 <- c(
    "--bfile", bfile_raw,
    "--keep", keep_overall,
    "--maf", maf_min,
    "--geno", geno_max,
    "--mind", mind_max,
    "--chr-set", chrset,
    "--make-bed", "--freq",
    "--out", bfile_qc_all
  )
  .plink_run(plink_bin, args1, must_exist = paste0(bfile_qc_all, ".bed"))
  
  # 3) Trait/system subset: keep only phenotyped FIDs among cleaned fam_keep
  dir.create(outdir_subset, recursive = TRUE, showWarnings = FALSE)
  phen_ring <- unique(as.character(ph_df[[ringnr_col]]))
  keep_pairs <- fam_keep[FID %in% phen_ring, .(FID, IID)]
  keep_subset <- file.path(outdir_subset, "keep.txt")
  fwrite(unique(keep_pairs), keep_subset, col.names = FALSE, sep = "\t")
  
  bfile_qc_sub <- file.path(outdir_subset, "qc")
  args2 <- c(
    "--bfile", bfile_qc_all,
    "--keep", keep_subset,
    "--maf", maf_min,
    "--geno", geno_max,
    "--mind", mind_max,
    "--chr-set", chrset,
    "--make-bed", "--freq",
    "--out", bfile_qc_sub
  )
  .plink_run(plink_bin, args2, must_exist = paste0(bfile_qc_sub, ".bed"))
  
  frq_file <- paste0(bfile_qc_sub, ".frq")
  list(bfile_qc_sub = bfile_qc_sub, frq_file = frq_file, keep_file = keep_subset)
}

# ------------------------------- --het ---------------------------------------
# Compute F on the subset bfile. Optionally pin to the same frequency file via --read-freq.

plink_run_het <- function(plink_bin,
                          bfile,
                          out_prefix,
                          chrset = 32,
                          autosomes_only = TRUE,
                          read_freq_file = NULL,
                          extra_args = character(0)) {
  args <- c(
    "--bfile", bfile,
    "--het",
    "--chr-set", chrset
  )
  if (autosomes_only) args <- c(args, "--autosome")
  if (!is.null(read_freq_file)) args <- c(args, "--read-freq", read_freq_file)
  if (length(extra_args)) args <- c(args, extra_args)
  args <- c(args, "--out", out_prefix)
  
  .plink_run(plink_bin, args, must_exist = paste0(out_prefix, ".het"))
  paste0(out_prefix, ".het")
}

# ------------------------- Read .het -> F_hat --------------------------------
read_plink_het <- function(het_file,
                           id_from = c("FID", "IID"),
                           center = TRUE,
                           scale = FALSE) {
  id_from <- match.arg(id_from)
  if (!file.exists(het_file)) stop("het_file not found: ", het_file)
  
  het <- data.table::fread(het_file, data.table = FALSE)
  
  nm <- names(het)
  
  if (all(c("FID","IID","O(HOM)","E(HOM)","N(NM)","F") %in% nm)) {
    # Fast path for your exact header
    ring   <- if (id_from == "FID") as.character(het$FID) else as.character(het$IID)
    O_HOM  <- suppressWarnings(as.numeric(het[["O(HOM)"]]))
    E_HOM  <- suppressWarnings(as.numeric(het[["E(HOM)"]]))
    N_SITE <- suppressWarnings(as.integer(het[["N(NM)"]]))
    F_raw  <- suppressWarnings(as.numeric(het[["F"]]))
  } else {
    # Fallback: normalize names then pick
    nm2 <- toupper(nm)
    nm2 <- gsub("[[:space:]]+", "", nm2)
    nm2 <- gsub("[()]", "", nm2)    # O(HOM)->O_HOM, N(NM)->N_NM
    nm2 <- gsub("[^A-Z0-9]+", "_", nm2)
    names(het) <- nm2
    
    req <- c("FID","IID","O_HOM","E_HOM","N_NM","F")
    miss <- setdiff(req, names(het))
    if (length(miss)) stop("PLINK .het missing columns after normalization: ", paste(miss, collapse=", "))
    
    ring   <- if (id_from == "FID") as.character(het$FID) else as.character(het$IID)
    O_HOM  <- suppressWarnings(as.numeric(het[["O_HOM"]]))
    E_HOM  <- suppressWarnings(as.numeric(het[["E_HOM"]]))
    N_SITE <- suppressWarnings(as.integer(het[["N_NM"]]))
    F_raw  <- suppressWarnings(as.numeric(het[["F"]]))
  }
  
  out <- data.frame(
    ringnr  = ring,
    O_HOM   = O_HOM,
    E_HOM   = E_HOM,
    N_SITES = N_SITE,
    F_hat   = F_raw,
    stringsAsFactors = FALSE
  )
  
  # Keep valid rows; center/scale F if requested
  out <- out[!is.na(out$ringnr) & !is.na(out$F_hat), , drop = FALSE]
  
  if (center || scale) {
    m <- if (center) mean(out$F_hat, na.rm = TRUE) else 0
    s <- if (scale) stats::sd(out$F_hat, na.rm = TRUE) else 1
    if (is.na(s) || s == 0) s <- 1
    out$F_hat <- if (center && !scale) out$F_hat - m else
      if (!center && scale)  out$F_hat / s else
        (out$F_hat - m) / s
  }
  
  out
}


# ------------------------ LMM + adjusted phenotype ---------------------------
fit_lmm_and_adjust <- function(dd, phenotype, include_F = TRUE) {
  stopifnot(phenotype %in% names(dd))
  base_fixed <- "sex + month + age"
  fixed <- if (include_F && "F_hat" %in% names(dd)) paste(base_fixed, "+ F_hat") else base_fixed
  
  form <- as.formula(paste0(
    phenotype, " ~ ", fixed, " + (1|ringnr) + (1|locality) + (1|hatch_year)"
  ))
  fit <- lmer(form, data = dd, control = lmerControl(optimizer = "Nelder_Mead"))
  
  re_list <- ranef(fit, condVar = FALSE)
  if (!"ringnr" %in% names(re_list)) stop("No random effect for ringnr found.")
  re_id <- as.data.frame(re_list$ringnr)
  colnames(re_id) <- "(Intercept)"
  re_id$ringnr <- rownames(re_list$ringnr)
  
  adj <- re_id %>%
    transmute(ringnr = as.character(ringnr), y_adjusted = `(Intercept)`)
  
  ind_stats <- dd %>%
    group_by(ringnr) %>%
    summarise(
      n_obs = dplyr::n(),
      y_mean = mean(.data[[phenotype]], na.rm = TRUE),
      .groups = "drop"
    )
  
  adj_phen <- adj %>% left_join(ind_stats, by = "ringnr")
  
  res_df <- data.frame(
    ringnr = dd$ringnr,
    y_obs  = dd[[phenotype]],
    resid  = resid(fit)
  )
  if ("F_hat" %in% names(dd)) res_df$F_hat <- dd$F_hat
  
  list(fit = fit, adj_phen = adj_phen, res_df = res_df)
}
