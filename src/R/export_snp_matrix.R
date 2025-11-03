# export_snp_matrix.R
# Exports a SNP matrix (rows = individuals, cols = selected SNPs) for Python.
# - Matches the phenotype/genotype alignment used earlier
# - Samples N individuals (from overlap) and K SNPs (randomly)
# - Optional mean-imputation of missing genotypes
# - Saves Feather (fast for Python) + CSV.gz + index files (ids and snp ids)

suppressPackageStartupMessages({
  library(bigsnpr)
  library(data.table)
  library(dplyr)
  library(arrow)    # for Feather (readable in Python with pandas/pyarrow)
})

# -------------------- CONFIG --------------------
plink_prefix <- "Data/combined_200k_70k_sparrow_genotype_data/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05"
adj_file     <- "Data/gnn/adjusted_body_mass.csv"  # from LMM step
out_prefix   <- "Data/gnn/snp_data"

n_indivs     <- 1000         # how many individuals to export
snp_count    <- NULL         # set to an integer to pick exactly that many SNPs (e.g., 20000)
snp_fraction <- 1.0          # or set a fraction in (0,1], e.g. 0.5 for 50% of SNPs
impute_mean  <- TRUE         # mean-impute missing calls (recommended)

# Optional: if you already have a fixed list of ringnr to use, provide a file:
ids_file <- "Data/grm_mass_helgeland_70K/grm.rel.id"  # a CSV with column 'ringnr'
# ------------------------------------------------

dir.create(dirname(out_prefix), recursive = TRUE, showWarnings = FALSE)

# --- Load phenotypes (LMM-adjusted) ---
adj <- fread(adj_file) %>% mutate(ringnr = as.character(ringnr))

# --- Load PLINK via bigsnpr ---
# creates .rds/.bk if not present
snp_readBed(paste0(plink_prefix, ".bed"), backingfile = plink_prefix)
obj <- snp_attach(paste0(plink_prefix, ".rds"))

G    <- obj$genotypes                    # FBM.code256
fam  <- obj$fam
map  <- obj$map
fam$FID <- as.character(fam$family.ID)   # ringnr
fam$IID <- as.character(fam$sample.ID)   # DNA ID (batch/plate)

# --- Deduplicate by ringnr (keep last), drop HIGHHET/MISSEX like your QC ---
fam_unique <- fam[!duplicated(fam$FID, fromLast = TRUE), ]
idx_unique <- match(fam_unique$IID, fam$IID)
G_unique   <- G[idx_unique, ]

mask_bad <- grepl("HIGHHET|MISSEX", fam_unique$IID)
if (any(mask_bad)) {
  fam_unique <- fam_unique[!mask_bad, ]
  G_unique   <- G_unique[!mask_bad, ]
}

cat("Individuals after dedup & QC:", nrow(fam_unique), "\n")

# --- Overlap with phenotyped ---
common_ids <- intersect(unique(adj$ringnr), fam_unique$FID)
cat("Phenotyped:", length(unique(adj$ringnr)),
    "Genotyped (unique):", nrow(fam_unique),
    "Overlap:", length(common_ids), "\n")

if (length(common_ids) == 0) stop("No overlap between adjusted phenotypes and genotypes.")

# --- Determine which individuals to export ---
if (!is.null(ids_file)) {
  id_tab <- fread(ids_file, header = FALSE)
  # Ensure columns exist
  if (ncol(id_tab) == 1) {
    setnames(id_tab, "V1")
  } else if (ncol(id_tab) >= 2) {
    setnames(id_tab, c("V1","V2"))
  }
  
  # fam_unique has the map FID (ringnr) <-> IID (DNA ID)
  # 1) Try assuming ids_file already contains ringnr (FID)
  cand1 <- intersect(as.character(id_tab$V1), common_ids)
  
  # 2) Otherwise, try mapping IIDs from ids_file -> ringnr via fam_unique
  map_IID_to_FID <- setNames(fam_unique$FID, fam_unique$IID)
  cand2 <- map_IID_to_FID[as.character(id_tab$V1)]
  if ("V2" %in% names(id_tab)) {
    cand2b <- map_IID_to_FID[as.character(id_tab$V2)]
    cand2 <- c(cand2, cand2b)
  }
  cand2 <- intersect(na.omit(unique(as.character(cand2))), common_ids)
  
  sel_ids <- unique(c(cand1, cand2))
  
  if (length(sel_ids) == 0) {
    stop("No IDs from ids_file overlap with genotype data after FID/IID mapping.")
  }
  cat("Using", length(sel_ids), "IDs from ids_file (after FID/IID alignment).\n")
} else {
  set.seed(42)
  take <- min(n_indivs, length(common_ids))
  sel_ids <- sample(common_ids, take)
  fwrite(data.frame(ringnr = sel_ids), paste0(out_prefix, "_selected_ids.csv"))
  cat("Sampled", length(sel_ids), "IDs; saved to ", paste0(out_prefix, "_selected_ids.csv"), "\n", sep = "")
}


# --- Subset to those individuals ---
keep_ind <- which(fam_unique$FID %in% sel_ids)
G_sub    <- G_unique[keep_ind, ]
id_order <- fam_unique$FID[keep_ind]     # order of rows in G_sub
stopifnot(length(id_order) == nrow(G_sub))

cat("Exporting", length(id_order), "individuals.\n")

# --- Choose SNPs (by count or fraction) ---
m_total <- ncol(G_sub)
if (!is.null(snp_count)) {
  k <- min(as.integer(snp_count), m_total)
} else {
  k <- min(as.integer(round(snp_fraction * m_total)), m_total)
}
set.seed(123)
snp_idx <- sort(sample.int(m_total, k))
cat("Selected", length(snp_idx), "SNPs out of", m_total, "\n")

# SNP metadata
snp_ids <- as.character(map$marker.ID)[snp_idx]
chr     <- map$chromosome[snp_idx]
pos     <- map$physical.pos[snp_idx]

# --- Extract dense matrix for the submatrix (individuals x selected SNPs) ---
# bigsnpr FBM.code256 can be indexed to return a base matrix when using [].
M <- G_sub[, snp_idx]  # numeric matrix with 0/1/2/NA

# --- Optional mean imputation of missing genotypes ---
if (impute_mean) {
  # Per SNP mean (ignoring NA). If all NA in a column, set mean = 0.
  col_means <- suppressWarnings(colMeans(M, na.rm = TRUE))
  col_means[!is.finite(col_means)] <- 0
  for (j in seq_len(ncol(M))) {
    nas <- is.na(M[, j])
    if (any(nas)) M[nas, j] <- col_means[j]
  }
}

# --- Build a data.frame for export: ringnr + SNP columns ---
# Beware: 65k columns is large but workable; feather/parquet handle this better than CSV.
geno_df <- data.frame(ringnr = id_order, M, check.names = FALSE)
names(geno_df) <- c("ringnr", snp_ids)

# --- Save outputs ---
# 1) Feather (fast, preserves types, great for Python/pandas with pyarrow)
feather_path <- paste0(out_prefix, "_geno.feather")
write_feather(geno_df, feather_path)
cat("Wrote Feather:", feather_path, "\n")

# 2) Also save as compressed CSV for easy inspection (large but universal)
csv_path <- paste0(out_prefix, "_geno.csv.gz")
fwrite(geno_df, csv_path)
cat("Wrote CSV.gz:", csv_path, "\n")

# 3) Save SNP metadata (ids, chr, pos) and the actual selected ids (rows)
fwrite(data.frame(snp_id = snp_ids, chr = chr, pos = pos),
       paste0(out_prefix, "_snp_meta.csv"))
fwrite(data.frame(ringnr = id_order),
       paste0(out_prefix, "_rows_order.csv"))

cat("Done.\n")