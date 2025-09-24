library(data.table)

# ---------- INPUT ----------
plink <- "\"C:/Users/Simen/OneDrive - NTNU/FYSMAT/INDMAT/25H/Prosjekt/PLINK/plink.exe\""
chrset <- 32
maf_min <- 0.01
geno_max <- 0.10
mind_max <- 0.05

pheno_file <- "Data/AdultMorphology_20240201_fix.csv"
bfile_raw  <- "Data/combined_200k_70k_sparrow_genotype_data/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05"
trait_col  <- "body_mass"

# Choose islands like in INLA example (set FALSE to keep all)
filter_isls <- TRUE
isls <- c(20, 22, 23, 24, 26, 27, 28, 33, 331, 332, 34, 35, 38)

out_prefix <- "Data/grm_mass_helgeland_70K/grm"
out_rds    <- "Data/gnn/GRM/GRM_vanraden_HELGELAND.rds"

# ---------- LOAD PHENOS ----------
ph <- fread(pheno_file)
ph[, ringnr := as.character(ringnr)]
if (filter_isls) ph <- ph[locality %in% isls]
ph <- ph[!is.na(get(trait_col))]

# ---------- PREP FAM (mirror INLA behavior) ----------
fam <- fread(paste0(bfile_raw, ".fam"), header = FALSE)
setnames(fam, c("V1","V2"), c("FID","IID"))   # FID=ringnr, IID=DNA ID (encoded)
fam[, FID := as.character(FID)]
fam[, IID := as.character(IID)]

# drop flagged samples
fam_keep <- fam[!grepl("HIGHHET|MISSEX", IID)]

# keep the *last* DNA ID per ringnr (as in the INLA code)
fam_keep <- fam_keep[!duplicated(fam_keep$FID, fromLast = TRUE)]

# ---------- 1) Overall QC ----------
dir.create("Data/qc_overall_", showWarnings = FALSE)
fwrite(fam_keep[, .(FID, IID)], "Data/qc_overall_/keep.txt",
       col.names = FALSE, sep = "\t")

cmd1 <- paste(
  plink,
  "--bfile", bfile_raw,
  "--keep Data/qc_overall_/keep.txt",
  "--maf", maf_min,
  "--geno", geno_max,
  "--mind", mind_max,
  "--chr-set", chrset,
  "--make-bed --freq",
  "--threads 8 --memory 48000",
  "--out Data/qc_overall_/qc"
)
stopifnot(system(cmd1) == 0)
bfile_qc_all <- "Data/qc_overall_/qc"

# ---------- 2) Trait/system subset QC ----------
dir.create("Data/qc_mass_helgeland", showWarnings = FALSE)

# Keep only phenotyped ringnr among *cleaned, deduplicated* fam_keep
keep_pairs <- fam_keep[FID %in% unique(ph$ringnr), .(FID, IID)]
fwrite(unique(keep_pairs), "Data/qc_mass_helgeland/keep.txt",
       col.names = FALSE, sep = "\t")

cmd2 <- paste(
  plink,
  "--bfile", bfile_qc_all,
  "--keep Data/qc_mass_helgeland/keep.txt",
  "--maf", maf_min,
  "--geno", geno_max,
  "--mind", mind_max,
  "--chr-set", chrset,
  "--make-bed --freq",
  "--threads 8 --memory 48000",
  "--out Data/qc_mass_helgeland/qc"
)
stopifnot(system(cmd2) == 0)
bfile_qc_sub <- "Data/qc_mass_helgeland/qc"
frq_file     <- "Data/qc_mass_helgeland/qc.frq"

# ---------- 3) Make GRM ----------
dir.create(dirname(out_prefix), recursive = TRUE, showWarnings = FALSE)
cmd3 <- paste(
  plink,
  "--bfile", bfile_qc_sub,
  "--keep Data/qc_mass_helgeland/keep.txt",
  "--maf", maf_min,
  "--geno", geno_max,
  "--mind", mind_max,
  "--chr-set", chrset,
  "--read-freq", frq_file,           # use the same freqs
  "--make-rel square bin cov",
  "--make-just-bim",                 # SNP list actually used
  "--threads 8 --memory 48000",
  "--out", out_prefix
)
stopifnot(system(cmd3) == 0)

# ---------- 4) Read + VanRaden scale ----------
rel_bin <- paste0(out_prefix, ".rel.bin")
rel_id  <- paste0(out_prefix, ".rel.id")
bim     <- paste0(out_prefix, ".bim")

id_tab <- fread(rel_id, header = FALSE)
ids <- id_tab$V1   # just ringnr (FID)
n <- length(ids)

M <- matrix(readBin(rel_bin, "numeric", n^2), nrow = n, byrow = TRUE)
rownames(M) <- ids
colnames(M) <- ids


snps   <- fread(bim, select = 2)
frq    <- fread(frq_file)
frq_inc <- frq[SNP %in% snps$V2]
stopifnot(nrow(frq_inc) > 0)

GRM <- M * (nrow(frq_inc) - 1) / (2 * sum(frq_inc$MAF * (1 - frq_inc$MAF)))

# ---------- SAVE ----------
dir.create(dirname(out_rds), recursive = TRUE, showWarnings = FALSE)
saveRDS(GRM, out_rds)
cat("Saved GRM with", n, "individuals to", out_rds, "\n")
