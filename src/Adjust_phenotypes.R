# ----- CONFIG -----
phenotype <- "thr_tarsus"  # set to "body_mass", "thr_tarsus", "thr_wing", etc.
infile <- "Data/AdultMorphology_20240201_fix.csv"
out_dir <- "Data/gnn"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
n_individuals <- NULL

# ----- LIBS -----
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(lme4)
})

# ----- LOAD + PREP -----
# The file uses semicolons in your header example; fread auto-detects but we'll be explicit.
dd <- fread(infile, sep = ";", data.table = FALSE)

# sanity check
req <- c("ringnr","adult_sex","year","month","day","locality","hatch_year","max_year",
         "first_locality","last_locality", "body_mass","thr_bill_depth","thr_bill_length",
         "thr_tarsus","thr_wing")
miss <- setdiff(req, names(dd))
if (length(miss)) stop("Missing columns: ", paste(miss, collapse=", "))

# keep only rows with non-missing phenotype
stopifnot(phenotype %in% colnames(dd))
dd <- dd[!is.na(dd[[phenotype]]), , drop = FALSE] %>%
  mutate(
    ringnr      = as.character(ringnr),
    sex         = factor(ifelse(adult_sex == 1, "m",
                                ifelse(adult_sex == 2, "f", NA))),
    month       = factor(as.integer(month)),
    locality    = factor(locality),
    hatch_year  = as.integer(as.character(hatch_year)),
    max_year    = as.integer(as.character(max_year)),
    age         = max_year - hatch_year
  ) %>%
  filter(!is.na(sex), !is.na(month), !is.na(age),
         !is.na(locality), !is.na(hatch_year))

message(sprintf("N individuals: %d; N observations: %d; phenotype: %s",
                dplyr::n_distinct(dd$ringnr), nrow(dd), phenotype))

# ----- LMM (two-step): fixed + random -----
# Fixed: sex + month + age
# Random: (1|ringnr) + (1|locality) + (1|hatch_year)
form <- as.formula(paste0(phenotype, " ~ sex + month + age + (1|ringnr) + (1|locality) + (1|hatch_year)"))

# Optimizer choices: Nelder_Mead is often robust for these models
fit <- lmer(form, data = dd, control = lmerControl(optimizer = "Nelder_Mead"))

# quick convergence check
if (!is.null(warnings())) {
  message("lmer emitted warnings (often harmless). Inspect if needed.")
}

# ----- EXTRACT ADJUSTED PHENOTYPE (BLUP for ringnr) -----
re_list <- ranef(fit, condVar = FALSE)
if (!"ringnr" %in% names(re_list)) stop("No random effect for ringnr found.")
re_id <- as.data.frame(re_list$ringnr)
colnames(re_id) <- "(Intercept)"
re_id$ringnr <- rownames(re_list$ringnr)
adj <- re_id %>%
  transmute(ringnr = as.character(ringnr),
            y_adjusted = `(Intercept)`)

# Also compute mean raw phenotype per individual and obs counts (useful for QA / later)
ind_stats <- dd %>%
  group_by(ringnr) %>%
  summarise(n_obs = dplyr::n(),
            y_mean = mean(.data[[phenotype]], na.rm = TRUE),
            .groups = "drop")

adj_phen <- adj %>% left_join(ind_stats, by = "ringnr")

# ----- (Optional) residuals at observation-level -----
res_df <- data.frame(
  ringnr = dd$ringnr,
  y_obs  = dd[[phenotype]],
  resid  = resid(fit)
)

# ----- SAVE -----
adj_out   <- file.path(out_dir, paste0("adjusted_", phenotype, ".csv"))
resid_out <- file.path(out_dir, paste0("residuals_", phenotype, "_obs.csv"))

write.csv(adj_phen, adj_out, row.names = FALSE)
write.csv(res_df,  resid_out, row.names = FALSE)

message("Saved: ", adj_out)
message("Saved: ", resid_out)

# ----- PRINT A TINY SUMMARY -----
message("Adjusted phenotype summary (first 6):")
print(head(adj_phen, 6))
