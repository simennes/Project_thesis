library(INLA)
library(dplyr)
library(data.table)

# File path to phenotype data
pheno_file <- "Data/AdultMorphology_20240201_fix.csv"

# File paths to genomic data
orig_geno_files <- paste0("Data/",
                          "combined_200k_70k_sparrow_genotype_data/",
                          "combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05",
                          c(".map", ".ped", ".fam", ".bim", ".bed"))

#### File path to plink program here
get_plink_path <- function() {
  "C:/Users/Simen/OneDrive - NTNU/FYSMAT/INDMAT/25H/Prosjekt/PLINK/plink.exe"
}



# Vector of terms in the INLA model
grm_effect_strs <- sapply(effects(grm = TRUE),
                          function(eff) make_effect_str(eff[[1]], eff[[2]]))

# Vector of ID codes (ring numbers) for genotyped individuals
genotyped_inds <- get_genotyped_inds(fam_file = orig_geno_files[3], sel = 1)

# Basic quality control of the genomic data
# Remove SNPs and individuals with too much mssing data, and too little genetic variation
qc_filt <- list(genorate_ind = 0.05,
                genorate_snp = 0.1,
                maf = 0.01)
qc_overall <- do_qc(fam_file = orig_geno_files[3],
                    ncores = 8,
                    mem = 8 * 6000,
                    qc_filt = qc_filt,
                    keep_inds = genotyped_inds,
                    sys = "",
                    resp = "overall")
genotyped_inds_qc <- get_genotyped_inds(fam_file = qc_overall[2], sel = 1)

# Example subsetting of the phenotype data
##### Just body mass measurements
response_colname <- "body_mass"
response <- "mass"
##### tarsus length:
# response_colname <- "thr_tarsus"
# response <- "tarsus"
##### wing length:
# response_colname <- "thr_wing"
# response <- "wing"
# Just Helgeland islands:
isls <- c(20, 22, 23, 24, 26, 27, 28, 33, 331, 332, 34, 35, 38)
sys_name = "helgeland"
####### Southern islands:
# isls <- c(60, 61, 63, 67, 68),
# sys_name <- "southern"
####### Helgeland and southern:
# isls <- c(20, 22, 23, 24, 26, 27, 28, 33, 331, 332, 34, 35, 38, 60, 61, 63, 67, 68)
# sys_name <- "merged"
# Create data frame of chosen phenotype data
pheno_data <- pheno_wrangle(filepath = pheno_file,
                            genotyped_inds = genotyped_inds_qc,
                            islands = isls,
                            y_col_name = response_colname,
                            testing = 100 # Use this to include only 100 obs, for fast testing of code
                            # testing = NULL # Use this to include all observations
                            )

# Create folds for a cross validation
cv_test_sets <- make_cv_test_sets(analysis_inds = unique(pheno_data$ringnr),
                                  num_folds = 10)

# Make new genomic data files that only includes the relevant phenotyped individuals
geno_files <- do_qc(fam_file = qc_overall[2],
                    ncores = 8,
                    mem = 8 * 6000,
                    qc_filt = qc_filt,
                    keep_inds = unique(pheno_data$ringnr),
                    sys = sys_name,
                    resp = response)
# Construct genomic relatedness matrix, and its inverse
grm_files <- make_raw_grm(analysis_inds = unique(pheno_data$ringnr),
                        bfile = gsub(".{4}$", "", geno_files[2]),
                        frq_file = geno_files[5],
                        genorate_ind = qc_filt$genorate_ind,
                        genorate_snp = qc_filt$genorate_snp,
                        ncores = 8,
                        mem = 8 * 6000,
                        maf = qc_filt$maf,
                        response = response,
                        geno_set = paste0(sys_name, "_70K"))
grm_obj <- compute_grm_obj(frq_file = geno_files[5],
                           rel_file = grm_files[2],
                           id_file = grm_files[1],
                           bim_file = grm_files[3],
                           pheno_data = pheno_data,
                           id_col = 1)

# Run a genomic prediction for one of the folds in the CV
# (you can make a for-loop do the full CV, but it will be very slow if using the full data set -
# I recommend using a computing cluster to do it in parallel)
gpgrm_cv <- run_gp(pheno_data = pheno_data,
                   train_inds = setdiff(unique(pheno_data$ringnr), cv_test_sets),
                   test_inds = cv_test_sets,
                   inverse_relatedness_matrix = grm_obj$inv_grm,
                   effects_vec = grm_effect_strs,
                   prior = pheno_data %>%
                     dplyr::filter(!ringnr %in% cv_test_sets) %>%
                     `$`("y") %>%
                     var() %>%
                     make_prior(pc_prec_upper_var = . / 2,
                                var_init = . / 3,
                                tau = 0.05))

n <- length(unique(pheno_data$ringnr))

# Extracting results from the INLA object

# Predicted breeding values (posterior means)
pred_bv <- gpgrm_cv$model$summary.random$id1$mean[order(gpgrm_cv$model$summary.random$id1$ID)][1:n]
# Posterior standard deviations
pred_bv_sd <- gpgrm_cv$model$summary.random$id1$sd[order(gpgrm_cv$model$summary.random$id1$ID)][1:n]
# Entries repeated to match repeated measurements
pheno_data$pred_bv_rep <- pred_bv[pheno_data$id1]
# Predicted phenotypes (posterior means)
pred_pheno <- gpgrm_cv$model$summary.fitted.values$mean
pred_pheno_sd <- gpgrm_cv$model$summary.fitted.values$sd

# Posterior statistics for variance components:
hyp_vars <- do.call(rbind,
                    lapply(gpgrm_cv$model$marginals.hyperpar,
                           inla_posterior_variances))
row.names(hyp_vars) <- gsub(pattern = "Precision",
                            replacement = "Variance",
                            x = row.names(hyp_vars))
hyp_vars

# Posterior statistics for fixed effects:
gpgrm_cv$model$summary.fixed
