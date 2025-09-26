# src/data.py
import os
import numpy as np
import pandas as pd
import logging

try:
    import pyreadr  # for .rds GRM files
except ImportError:
    # pyreadr is only needed when reading .rds GRM; allow import to fail if not used
    pyreadr = None


def load_data(paths, target_column: str = "y_adjusted", standardize_features: bool = False):
    """
    Load SNP, phenotype, and optionally GRM; support fast path via NPZ.

    paths can contain either:
      - 'npz': path to prepacked npz with keys {'snp', 'body_mass', optional 'ids'}
        and optionally 'grm_rds' if GRM-based graphs are needed.
      - or the raw files: {'snp_feather', 'phenotype_csv', optional 'grm_rds'}

    Returns: (X:np.ndarray, y:np.ndarray, ids:np.ndarray[str] or None, GRM_df:pd.DataFrame or None)
    """
    npz_path = paths.get("npz")
    X: np.ndarray
    y: np.ndarray
    ids = None
    GRM_df = None

    if npz_path and os.path.exists(npz_path):
        logging.info(f"Loading data from NPZ: {npz_path}")
        data = np.load(npz_path, allow_pickle=False)
        if "snp" not in data or "body_mass" not in data:
            raise ValueError("NPZ must contain 'snp' and 'body_mass' arrays.")
        X = data["snp"].astype(np.float32, copy=False)
        y = data["body_mass"].astype(np.float32, copy=False)
        if "ids" in data:
            ids = data["ids"].astype(str)
    else:
        # --- SNPs ---
        logging.info(f"Loading SNP data from feather: {paths['snp_feather']}")
        snp = pd.read_feather(paths["snp_feather"])
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

        # Intersect keeping SNP order
        ids_snp = snp.index.to_numpy()
        mask = np.isin(ids_snp, pheno.index.to_numpy())
        ids_common = ids_snp[mask]
        snp = snp.loc[ids_common]
        pheno = pheno.loc[ids_common]
        ids = ids_common

        # Sanity checks
        if snp.isna().any().any() or pheno[target_column].isna().any():
            raise ValueError("Found NaNs after alignment. Check input files.")
        X = snp.values.astype(np.float32)
        y = pheno[target_column].values.astype(np.float32)

    # Optional GRM
    grm_rds = paths.get("grm_rds")
    if grm_rds:
        if pyreadr is None:
            raise ImportError("pyreadr is required to read .rds GRM files. pip install pyreadr")
        logging.info(f"Loading GRM from RDS: {grm_rds}")
        grm_res = pyreadr.read_r(grm_rds)
        GRM_df = grm_res[None]
        GRM_df.index = GRM_df.index.astype(str)
        GRM_df.columns = GRM_df.columns.astype(str)
        if ids is not None:
            # Align GRM to ids order (drop missing)
            ids_in = np.array([i for i in ids if i in GRM_df.index], dtype=str)
            GRM_df = GRM_df.loc[ids_in, ids_in]
            # Realign X/y/ids too, to ensure consistent order with GRM if needed later
            if len(ids_in) != len(ids):
                # filter X/y by ids_in order
                idx_map = {v: k for k, v in enumerate(ids)}
                sel = np.array([idx_map[v] for v in ids_in], dtype=int)
                X = X[sel]
                y = y[sel]
                ids = ids_in

    if standardize_features:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    return X, y, ids, GRM_df
