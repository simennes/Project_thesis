# ===== NEW: Genomic inbreeding from a .feather genotype file =====
# Requires: install.packages("arrow")
suppressPackageStartupMessages(library(arrow))

geno_feather_file <- "Data/gnn/SNP/ALL/snp_export_body_mass_ALL_geno.feather"
geno_id_cols_try  <- c("ringnr", "FID", "IID", "id", "individual")  # auto-detect id col

maf_min      <- 0.01
callrate_min <- 0.95
center_F     <- TRUE
scale_F      <- FALSE

compute_F_from_feather <- function(path,
                                   id_col = NULL,
                                   maf_min = 0.01,
                                   callrate_min = 0.95) {
  if (!file.exists(path)) stop("Feather file not found: ", path)
  
  tbl <- read_feather(path)
  # Try to guess long vs wide
  nm <- names(tbl)
  
  # Try to pick an ID column if not provided
  if (is.null(id_col)) {
    hit <- intersect(geno_id_cols_try, nm)
    if (length(hit) == 0) stop("Could not find an ID column. Present columns: ", paste(nm, collapse=", "))
    id_col <- hit[1]
  }
  tbl[[id_col]] <- as.character(tbl[[id_col]])
  
  # Heuristic: long format if it has columns like (id, snp, geno) or (id, variant, dosage)
  long_candidates <- list(
    c(id_col, "snp", "geno"),
    c(id_col, "SNP", "GENO"),
    c(id_col, "marker", "geno"),
    c(id_col, "variant", "dosage"),
    c(id_col, "snp_id", "dosage"),
    c(id_col, "snp", "dosage"),
    c(id_col, "marker", "dosage")
  )
  is_long <- FALSE
  which_long <- NULL
  for (cand in long_candidates) {
    if (all(cand %in% nm)) { is_long <- TRUE; which_long <- cand; break }
  }
  
  if (is_long) {
    # ---- LONG FORMAT: columns: id_col, snp_col, geno_col (0/1/2 or NA)
    snp_col  <- which_long[2]
    geno_col <- which_long[3]
    # Ensure numeric and NA for missing codes
    tbl[[geno_col]] <- suppressWarnings(as.numeric(tbl[[geno_col]]))
    # QC per SNP
    snp_stats <- tbl |>
      dplyr::group_by(.data[[snp_col]]) |>
      dplyr::summarise(
        callrate = mean(!is.na(.data[[geno_col]])),
        p = mean(.data[[geno_col]], na.rm = TRUE)/2,
        .groups = "drop"
      ) |>
      mutate(maf = pmin(p, 1 - p))
    
    keep_snps <- snp_stats |>
      filter(is.finite(p), p > 0, p < 1, callrate >= callrate_min, maf >= maf_min) |>
      pull(.data[[snp_col]])
    
    if (length(keep_snps) == 0) stop("After QC, no SNPs remain (long format). Relax thresholds or check data.")
    
    snp_stats <- snp_stats |>
      filter(.data[[snp_col]] %in% keep_snps) |>
      mutate(E = 2 * p * (1 - p)) |>
      select(all_of(snp_col), E)
    
    # Merge E back to compute Hexp per (id,snp)
    tbl_keep <- tbl |>
      filter(.data[[snp_col]] %in% keep_snps) |>
      left_join(snp_stats, by = setNames(snp_col, snp_col))
    
    # Per individual: Hobs = frac(geno==1), Hexp = mean(E over non-missing)
    per_id <- tbl_keep |>
      group_by(.data[[id_col]]) |>
      summarise(
        het_count = sum(.data[[geno_col]] == 1, na.rm = TRUE),
        nonmiss   = sum(!is.na(.data[[geno_col]])),
        Hobs      = het_count / nonmiss,
        Hexp      = mean(E[!is.na(.data[[geno_col]])], na.rm = TRUE),
        .groups = "drop"
      )
    
  } else {
    # ---- WIDE FORMAT: one row per individual, SNP columns are 0/1/2 or NA
    # Guess SNP columns: numeric-like with many columns
    non_id_cols <- setdiff(nm, id_col)
    # Keep columns that look numeric (or can be coerced) and are many
    numeric_like <- vapply(tbl[non_id_cols], function(x) {
      is.numeric(x) || is.integer(x) || is.double(x)
    }, logical(1))
    snp_cols <- non_id_cols[numeric_like]
    if (length(snp_cols) < 50) {
      stop("Wide format detected but found too few numeric SNP columns. Found: ", length(snp_cols))
    }
    # Coerce to numeric, treat -9 as NA
    for (cc in snp_cols) {
      x <- tbl[[cc]]
      x[x %in% c(-9, "-9")] <- NA
      tbl[[cc]] <- suppressWarnings(as.numeric(x))
    }
    
    G <- as.matrix(tbl[, snp_cols, drop = FALSE])
    # SNP QC
    callrate <- colMeans(!is.na(G))
    p_vec    <- colMeans(G, na.rm = TRUE)/2
    maf      <- pmin(p_vec, 1 - p_vec)
    keep     <- (callrate >= callrate_min) & (maf >= maf_min) & is.finite(p_vec) & p_vec > 0 & p_vec < 1
    if (!any(keep)) stop("After QC, no SNPs remain (wide format). Relax thresholds or check data.")
    G  <- G[, keep, drop = FALSE]
    p  <- colMeans(G, na.rm = TRUE)/2
    E  <- 2 * p * (1 - p)
    
    het_counts <- rowSums(G == 1, na.rm = TRUE)
    nonmiss    <- rowSums(!is.na(G))
    Hobs       <- het_counts / nonmiss
    
    nonmiss_ind <- (!is.na(G)) * 1.0
    Hexp_sum    <- as.vector(nonmiss_ind %*% E)
    Hexp        <- Hexp_sum / nonmiss
    
    per_id <- data.frame(
      !!id_col := tbl[[id_col]],
      het_count = het_counts,
      nonmiss   = nonmiss,
      Hobs      = Hobs,
      Hexp      = Hexp,
      check.names = FALSE
    )
  }
  
  per_id$Hexp[!is.finite(per_id$Hexp) | per_id$Hexp < 1e-10] <- NA
  per_id <- per_id |> filter(!is.na(Hexp) & nonmiss > 0)
  
  per_id <- per_id |>
    mutate(F_hat = 1 - (Hobs / Hexp)) |>
    select(all_of(id_col), F_hat) |>
    distinct()
  
  names(per_id)[1] <- "ringnr"   # unify name for merging
  per_id$ringnr <- as.character(per_id$ringnr)
  per_id
}

# Compute F_hat and merge into dd
F_df <- compute_F_from_feather(
  path = geno_feather_file,
  id_col = NULL,                 # set manually if auto-detect fails, e.g. "ringnr"
  maf_min = maf_min,
  callrate_min = callrate_min
)

# Optional centering/scaling
if (nrow(F_df) == 0) stop("No F_hat values computed (after QC/merging).")
if (center_F || scale_F) {
  mm <- mean(F_df$F_hat, na.rm = TRUE); ss <- sd(F_df$F_hat, na.rm = TRUE)
  if (center_F && !scale_F)      F_df$F_hat <- F_df$F_hat - mm
  else if (center_F && scale_F)  F_df$F_hat <- (F_df$F_hat - mm)/ifelse(ss > 0, ss, 1)
  else if (!center_F && scale_F) F_df$F_hat <- F_df$F_hat/ifelse(ss > 0, ss, 1)
}

dd <- dd %>% left_join(F_df, by = "ringnr") %>% filter(!is.na(F_hat))
message(sprintf("Computed F_hat for %d individuals; merged rows in dd: %d",
                nrow(F_df), nrow(dd)))

# ---- Update the model to include F_hat as fixed effect (rest of your script stays the same) ----
form <- as.formula(paste0(
  phenotype, " ~ sex + month + age + F_hat + (1|ringnr) + (1|locality) + (1|hatch_year)"
))
fit <- lmer(form, data = dd, control = lmerControl(optimizer = "Nelder_Mead"))
