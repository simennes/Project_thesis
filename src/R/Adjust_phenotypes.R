# make_adjusted_phenotypes_plink.R
# QC+subset with PLINK, compute --het (F_hat), merge into LMM, export adjusted phenotypes.

# ------------------------------ CONFIG ---------------------------------------
phenotype   <- "thr_wing"                    # e.g. "body_mass", "thr_tarsus", "thr_wing"
infile      <- "../../Data/AdultMorphology_20240201_fix.csv"
out_dir     <- "../../Data/gnn"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# PLINK
plink_bin   <- "C:/Users/Simen/OneDrive - NTNU/FYSMAT/INDMAT/25H/Prosjekt/PLINK/plink.exe"
bfile_raw   <- "../../Data/combined_200k_70k_sparrow_genotype_data/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05"

# QC thresholds (same spirit as your GRM script)
chrset      <- 32
maf_min     <- 0.01
geno_max    <- 0.10
mind_max    <- 0.05

# Optional: restrict phenotypes to certain islands (like your INLA example)
filter_isls <- FALSE
isls        <- c(20,22,23,24,26,27,28,33,331,332,34,35,38)

# Which PLINK ID equals 'ringnr' in phenotypes? (usually FID)
id_from_plink <- "FID"   # or "IID"

# Center/scale F for interpretability
center_F <- TRUE
scale_F  <- FALSE

# Optional extra PLINK args for --het (leave empty usually)
extra_het_args <- character(0)
# Example if you *really* want to pin allele freqs:
# extra_het_args <- c("--read-freq", "Data/qc_subset/qc.frq")

# Support
source("adjust_support.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lme4)
})

# ---------------------------- LOAD PHENOTYPES --------------------------------
dd <- fread(infile, sep = ";", data.table = FALSE)


req <- c("ringnr","adult_sex","year","month","day","locality","hatch_year","max_year",
         "first_locality","last_locality","body_mass","thr_bill_depth","thr_bill_length",
         "thr_tarsus","thr_wing")
miss <- setdiff(req, names(dd))
if (length(miss)) stop("Missing columns: ", paste(miss, collapse=", "))

stopifnot(phenotype %in% colnames(dd))

dd <- dd[!is.na(dd[[phenotype]]), , drop = FALSE] |>
  mutate(
    ringnr      = as.character(ringnr),
    sex         = factor(ifelse(adult_sex == 1, "m",
                                ifelse(adult_sex == 2, "f", NA))),
    month       = factor(as.integer(month)),
    locality    = factor(locality),
    hatch_year  = as.integer(as.character(hatch_year)),
    max_year    = as.integer(as.character(max_year)),
    age         = max_year - hatch_year
  ) |>
  filter(!is.na(sex), !is.na(month), !is.na(age),
         !is.na(locality), !is.na(hatch_year))

if (filter_isls) {
  dd <- dd |> filter(locality %in% isls)
}

message(sprintf("N individuals: %d; N observations: %d; phenotype: %s",
                dplyr::n_distinct(dd$ringnr), nrow(dd), phenotype))

# ------------------------- QC + SUBSET GENOTYPES -----------------------------
qc_dirs <- list(
  overall = file.path(out_dir, "qc_overall_"),
  subset  = file.path(out_dir, "qc_subset")
)

qc_res <- plink_qc_and_subset(
  plink_bin     = plink_bin,
  bfile_raw     = bfile_raw,
  ph_df         = dd,
  ringnr_col    = "ringnr",
  outdir_overall= qc_dirs$overall,
  outdir_subset = qc_dirs$subset,
  maf_min       = maf_min,
  geno_max      = geno_max,
  mind_max      = mind_max,
  chrset        = chrset,
  drop_iid_regex= "HIGHHET|MISSEX"
)

bfile_qc_sub <- qc_res$bfile_qc_sub
frq_file     <- qc_res$frq_file

# ------------------------------- --het (F) -----------------------------------
plink_out <- file.path(out_dir, "plink_het_subset")
het_path <- plink_run_het(
  plink_bin      = plink_bin,
  bfile          = bfile_qc_sub,
  out_prefix     = plink_out,
  chrset         = chrset,
  autosomes_only = TRUE,
  # If you want --read-freq, uncomment next line to pin to the subset's .frq:
  # read_freq_file = frq_file,
  extra_args     = extra_het_args
)

F_df <- read_plink_het(
  het_file = het_path,
  id_from  = id_from_plink,
  center   = center_F,
  scale    = scale_F
)

# Merge and keep rows with F
dd <- dd %>% left_join(F_df, by = "ringnr") %>% filter(!is.na(F_hat))
message(sprintf("Merged F_hat for %d individuals; rows in dd: %d",
                nrow(F_df), nrow(dd)))

# -------------------------- FIT + EXTRACT ADJUSTED ---------------------------
fit_res  <- fit_lmm_and_adjust(dd, phenotype = phenotype, include_F = TRUE)
fit      <- fit_res$fit

loc_df <- dd %>%
  dplyr::distinct(ringnr, locality) %>%
  dplyr::mutate(locality = as.character(locality))  # write readable codes to CSV

adj_phen <- fit_res$adj_phen %>%
  dplyr::left_join(F_df,  by = "ringnr") %>%
  dplyr::left_join(loc_df, by = "ringnr")  # <-- adds 'locality' column

# keep last row per ringnr in the existing order of adj_phen
adj_phen <- adj_phen[!duplicated(adj_phen$ringnr, fromLast = TRUE), ]

# -------------------------------- SAVE ---------------------------------------
adj_out    <- file.path(out_dir, paste0("adjusted_", phenotype, ".csv"))
write.csv(adj_phen, adj_out, row.names = FALSE)
message("Saved: ", adj_out)


# ------------------------------ SUMMARY --------------------------------------
message("\nFixed-effects summary:")
print(summary(fit)$coefficients)

message("\nAdjusted phenotype summary (first 6):")
print(utils::head(adj_phen, 6))

# any duplicates?
dups <- adj_phen$ringnr[duplicated(adj_phen$ringnr)]
dups
length(dups)
