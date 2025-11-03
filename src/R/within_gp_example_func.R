effects <- function(grm) {

  l <- list(list("1", "fixed"),
            list("sex", "fixed"),
            list("month", "fixed"),
            list("age", "fixed"),
            list("hatch_year", "iid"),
            list("island", "iid"),
            list("id2", "iid"),
            list("day_session", "iid"))

  if (grm) {
    l <- c(l, list(list("id1", "grm")))
  } else if (rr) {
    l <- c(l, list(list("id1", "rr", "prior$rr_effect_var")))
  }

  l
}

make_effect_str <- function(name,
                            type,
                            hyp = "prior$hyperpar_var",
                            custom_str = NULL,
                            inv_grm = "inverse_relatedness_matrix") {
  switch(type,
         fixed = name,
         iid = paste0("f(", name, ", model = \"iid\", hyper = ", hyp, ")"),
         rr = paste0("f(",
                     name,
                     ", model = \"z\", Z = pc_matrix, hyper = ",
                     hyp,
                     ")"),
         grm = paste0("f(",
                      name,
                      ", values = as.numeric(colnames(",
                      inv_grm,
                      ")), model = \"generic0\", hyper = ",
                      hyp,
                      ", constr = FALSE, Cmatrix = ",
                      inv_grm,
                      ")"),
         custom = custom_str
  )
}

# gets iid from .fam
get_genotyped_inds <- function(fam_file,
                               sel = 2 # 1: fid; 2: iid
)  {
  fread(fam_file, select = sel, data.table = FALSE, header = FALSE)[, 1]
}

do_qc <- function(fam_file,
                  ncores,
                  mem,
                  qc_filt,
                  keep_inds,
                  sys,
                  resp) {

  file_root <- gsub(pattern = ".fam", replacement = "", x = fam_file)
  dir <- paste0("Data/qc", "_", resp, "_", sys)
  fam <- fread(fam_file, select = c(1, 2), data.table = FALSE, header = FALSE)
  fam_keep <- fam[fam$V1 %in% keep_inds, ]
  # Exclude samples with high heterozygosity
  fam_keep <- fam_keep[!grepl(pattern = ".*HIGHHET.*", x = fam_keep$V2), ]
  # Exclude samples with mismatches in sex
  fam_keep <- fam_keep[!grepl(pattern = ".*MISSEX.*", x = fam_keep$V2), ]

  # For inds. genotyped multiple times, keep last one
  fam_keep <- fam_keep[!duplicated(fam_keep$V1, fromLast = TRUE), ]

  dir.create(dir, showWarnings = FALSE)
  write.table(fam_keep,
              file = paste0(dir, "/keep.txt"),
              quote = FALSE,
              row.names = FALSE,
              col.names = FALSE)

  exit_code <-
    system2(get_plink_path(),
            paste0("--bfile ", file_root, " ",
                   "--make-bed ",
                   "--freq ",
                   "--maf ", qc_filt$maf, " ", # Filter by minor allele frequency
                   "--geno ", qc_filt$genorate_snp, " ", # Filter SNPs by call rate
                   "--mind ", qc_filt$genorate_ind, " ", # Filter inds by call rate
                   "--chr-set 32 ", # Sparrow chromosomes
                   "--memory ", mem, " ",
                   "--keep ", dir, "/keep.txt ",
                   "--threads ", ncores, " ",
                   "--out ", dir, "/qc"))

  if (exit_code != 0) {
    stop("Error in plink")
  }

  paste0(dir, "/", c("keep.txt", paste0("qc.", c("fam", "bim", "bed", "frq"))))

}

pheno_wrangle <- function(filepath,
                          genotyped_inds,
                          islands,
                          y_col_name,
                          testing = NULL) {
  # Load data
  dat <- fread(file = filepath, header = TRUE, data.table = FALSE) %>%
    # Keep only the...
    dplyr::filter(ringnr %in% genotyped_inds) %>% # genotyped individuals,
    dplyr::filter(!is.na(get(y_col_name))) %>% # phenotyped individuals,
    dplyr::filter(locality %in% islands) # and measurements in the system
  # Merge Lurøy and Onøy
  dat$first_locality <- ifelse(dat$first_locality %in% c(331, 332),
                               33,
                               dat$first_locality)

  # Do some checks
  stopifnot(all(dat$adult_sex %in% c(1, 2)))

  df <- data.frame(ringnr = dat$ringnr,
                   sex = factor(ifelse(dat$adult_sex == 1, "m", "f")),
                   y = dat[, y_col_name],
                   year = factor(dat$year),
                   month = factor(dat$month),
                   island = factor(dat$locality),
                   hatch_year = factor(dat$hatch_year),
                   first_island = factor(dat$first_locality),
                   age = dat$year - dat$hatch_year,
                   day_session = interaction(dat$ringnr,
                                             dat$year,
                                             dat$month,
                                             dat$day,
                                             drop = TRUE),
                   id1 = match(dat$ringnr, unique(dat$ringnr)),
                   id2 = match(dat$ringnr, unique(dat$ringnr)))

  if (!is.null(testing)) {
    s <- sort(sample(dim(df)[1], testing))
    while (any(summary(df$island[s]) == 0)) {
      s <- sample(dim(df)[1], testing)
    }
    df <- df[s, ]
    df$id1 <- df$id2 <- match(df$ringnr, unique(df$ringnr))
  }

  df
}

make_raw_grm <- function(analysis_inds,
                         bfile,
                         frq_file,
                         test_islands = NULL,
                         train_islands = NULL,
                         ncores,
                         mem,
                         genorate_ind,
                         genorate_snp,
                         maf,
                         response,
                         geno_set,
                         rel_cutoff = 1 - 1e-8) {

  dir <- paste0("Data/grm_", response, "_", geno_set)

  if (!is.null(train_islands)) {
    dir <- paste0(dir, "_train", train_islands$code)
  }
  if (!is.null(test_islands)) {
    dir <- paste0(dir, "_test", test_islands$code)
  }

  dir.create(dir, showWarnings = FALSE)

  fam <- fread(file = paste0(bfile, ".fam"), header = FALSE)

  write.table(fam[fam$V1 %in% analysis_inds, ],
              file = paste0(dir, "/keep.txt"),
              quote = FALSE,
              row.names = FALSE,
              col.names = FALSE)

  exit_code <- system2(get_plink_path(),
                       paste0("--bfile ", bfile, " ",
                              "--maf ", maf, " ",
                              "--keep ", dir, "/keep.txt ",
                              "--geno ", genorate_snp, " ",
                              "--mind ", genorate_ind, " ",
                              "--chr-set 32 ",
                              "--memory ", mem, " ",
                              "--read-freq ", frq_file, " ",
                              "--rel-cutoff ", rel_cutoff, " ",
                              "--make-rel square bin cov ", # calculate raw GRM
                              "--make-just-bim ", # to know which SNPS were kept
                              "--threads ", ncores, " ",
                              "--out ", dir, "/grm"))

  if (exit_code != 0) {
    stop("Error in plink")
  }

  paste0(dir, "/grm.", c("rel.id", "rel.bin", "bim"))
}

compute_grm_obj <- function(frq_file,
                            rel_file,
                            id_file,
                            bim_file,
                            pheno_data,
                            id_col) {

  snps <- fread(file = bim_file, select = 2, data.table = FALSE, header = FALSE)
  frq <- fread(file = frq_file, select = c("SNP", "MAF"), header = TRUE)
  frq_inc <- frq[frq$SNP %in% snps$V2]
  ids <- fread(file = id_file,
               select = id_col,
               data.table = FALSE,
               header = FALSE)[, 1]
  n <- length(ids)

  vr_grm <- matrix(readBin(rel_file,
                           "numeric",
                           n ^ 2),
                   nrow = n) *
    (dim(frq_inc)[1] - 1) / # convert from sample covariance
    (2 * sum(frq_inc$MAF * (1 - frq_inc$MAF))) # vanRaden scaling

  # Rename columns for INLA
  dimnames(vr_grm)[[1]] <- dimnames(vr_grm)[[2]] <-
    pheno_data[match(ids, pheno_data$ringnr), "id1"]

  # Check symmetry
  if (!isSymmetric(vr_grm))
    stop("GRM not symmetric")

  # Compute eigenvalues
  e_vals <- eigen(vr_grm, only.values = TRUE)$values
  # Add small value to diagonal to get pos. def. matrix
  add_val <- ifelse(tail(e_vals, 1) < 0, - tail(e_vals, 1), 0) + 1e-9
  vr_grm <- vr_grm + diag(add_val, nrow(vr_grm))

  # Check positive defitiveness:
  e_vals_new <- eigen(vr_grm, only.values = TRUE)$values
  if (!all(Im(e_vals_new) == 0 & Re(e_vals_new) > 0))
    stop("GRM not positive definite")

  list(grm = vr_grm, inv_grm = solve(vr_grm), add_val = add_val)
}

make_prior <- function(pc_matrix = NULL,
                       va_apriori = NULL,
                       # more accurately, a priori V_a over variance in data.
                       pc_prec_upper_var,
                       var_init,
                       var_init_id = var_init,
                       tau = 0.05,
                       pc_prec_upper_var_id = pc_prec_upper_var) {

  rr_effect_var <- NULL
  if (!is.null(pc_matrix) && !is.null(va_apriori)) {
    # Assume each PC-"marker" has N(0, var), where var is the
    # PC-equivalent of va/2sum(p(1-p))
    rr_effect_var <- list(
      prec = list(fixed = TRUE,
                  initial = log(1 / (va_apriori / sum(diag(var(pc_matrix))))))
    )
  }
  # PC prior for identity effect
  hyperpar_var_id <- list(
    prec = list(initial = log(1 / var_init_id),
                prior = "pc.prec",
                param = c(sqrt(pc_prec_upper_var_id), tau),
                fixed = FALSE))
  # PC priors for other random effects
  hyperpar_var <- list(
    prec = list(initial = log(1 / var_init),
                prior = "pc.prec",
                param = c(sqrt(pc_prec_upper_var), tau),
                fixed = FALSE))

  list(rr_effect_var = rr_effect_var,
    hyperpar_var_id = hyperpar_var_id,
    hyperpar_var = hyperpar_var)
}

make_cv_test_sets <- function(analysis_inds,
                              num_folds = 10) {
  n <- length(analysis_inds)

  # Split individuals into n folds
  scrambled <- sample(n)
  folds <- cut(seq_len(n), num_folds)
  levels(folds) <- seq_len(num_folds)

  # Create list of test individuals in each fold
  lapply(1:num_folds, function(fo) {
    x <- analysis_inds[sort(scrambled[folds == fo])]
    attributes(x) <- list(fo = fo)
    x
  })
}

run_gp <- function(pheno_data,
                   train_inds,
                   test_inds, # The rest in analysis_inds are used for training
                   pc_matrix = NULL,
                   inverse_relatedness_matrix = NULL,
                   effects_vec,
                   prior,
                   ncores = tar_option_get("resources")$clustermq$template$cores
) {

  n_test <- length(test_inds)
  n_train <- length(train_inds)
  n <- n_test + n_train

  inla_formula <- stats::reformulate(effects_vec, response = "y_na")

  pheno_data$y_na <- pheno_data$y
  pheno_data$y_na[pheno_data$ringnr %in% test_inds] <- NA

  model <- INLA::inla(inla_formula,
                      family = "gaussian",
                      data = pheno_data,
                      verbose = TRUE,
                      control.compute = list(config = TRUE),
                      control.family = list(hyper = prior$hyperpar_var))
  model <- INLA::inla.rerun(model)
  model <- INLA::inla.rerun(model)

  samp <- INLA::inla.posterior.sample(n = 1e4,
                                      result = model,
                                      add.names = FALSE)

  # Don't save some large things we don't need
  model$misc$configs$config <-
    model$.args <-
    model$call <-
    model$all.hyper <-
    NULL
  list(model = model, samp = samp)
}

inla_posterior_variances <- function(prec_marginal) {
  sigma_marg <- INLA::inla.tmarginal(function(x) 1 / x, prec_marginal)
  INLA::inla.zmarginal(sigma_marg, silent = TRUE)
}
