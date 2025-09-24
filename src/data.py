# src/data.py
import os
import numpy as np
import pandas as pd

try:
    import pyreadr  # for .rds GRM files
except ImportError as e:
    raise ImportError("pyreadr is required to read .rds GRM files. pip install pyreadr") from e


def load_data(paths, target_column="y_adjusted", standardize_features=False):
    """
    Load SNP, phenotype, and GRM; intersect by ringnr; align to GRM order.
    paths: dict with keys ['snp_feather', 'phenotype_csv', 'grm_rds']
    returns: (X:np.ndarray, y:np.ndarray, ids:np.ndarray[str], GRM_df:pd.DataFrame)
    """
    # --- SNPs ---
    snp = pd.read_feather(paths["snp_feather"])
    # expect first column ringnr; if not, try column name
    if "ringnr" not in snp.columns:
        raise ValueError("SNP feather must contain a 'ringnr' column as first identifier column.")
    snp = snp.set_index("ringnr")
    snp.index = snp.index.astype(str)

    # --- Phenotypes ---
    pheno = pd.read_csv(paths["phenotype_csv"])
    if "ringnr" not in pheno.columns:
        raise ValueError("Phenotype CSV must contain a 'ringnr' column.")
    if target_column not in pheno.columns:
        raise ValueError(f"Phenotype CSV must contain the target column '{target_column}'.")
    pheno = pheno.set_index("ringnr")
    pheno.index = pheno.index.astype(str)

    # --- GRM ---
    grm_res = pyreadr.read_r(paths["grm_rds"])
    GRM_df = grm_res[None]  # dataframe with row & col names as IDs
    GRM_df.index = GRM_df.index.astype(str)
    GRM_df.columns = GRM_df.columns.astype(str)

    # --- Intersect and preserve GRM order (as in your notebook) ---
    ids_grm = GRM_df.index.to_numpy()
    ids_snp = snp.index.to_numpy()
    ids_ph = pheno.index.to_numpy()
    mask_in = np.isin(ids_grm, ids_snp) & np.isin(ids_grm, ids_ph)
    ids_common = ids_grm[mask_in]  # GRM order preserved

    # reindex SNP & pheno to this exact order
    snp = snp.reindex(ids_common)
    pheno = pheno.reindex(ids_common)
    GRM_df = GRM_df.loc[ids_common, ids_common]

    # Sanity checks
    if snp.isna().any().any() or pheno[target_column].isna().any():
        raise ValueError("Found NaNs after alignment. Check input files.")
    X = snp.values.astype(np.float32)
    y = pheno[target_column].values.astype(np.float32)
    ids = ids_common

    if standardize_features:
        # Optional standardization; SNPs are typically 0/1/2 so usually not needed
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    return X, y, ids, GRM_df
