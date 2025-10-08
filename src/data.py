# src/data.py
import os
import numpy as np
import pandas as pd
import pyreadr  # for .rds GRM files
import logging

def load_data(paths, target_column: str = "y_adjusted", standardize_features: bool = False):
    """
    Load SNP, phenotype, and optionally GRM; supports fast path via NPZ while
    STILL loading/aligning the GRM from RDS when requested.

    paths may contain either:
      - 'npz' (or 'npz_path'): path to NPZ with keys {'snp','body_mass', optional 'ids'}
        and optionally 'grm_rds' for GRM-based graphs.
      - or raw files: {'snp_feather','phenotype_csv', optional 'grm_rds'}

    Returns:
      X: np.ndarray [n, p] float32
      y: np.ndarray [n]     float32
      ids: np.ndarray[str] (or None)
      GRM_df: pd.DataFrame square [n, n] (or None)
    """

    # ---- Inputs / defaults
    X: np.ndarray
    y: np.ndarray
    ids = None
    GRM_df = None

    # Accept both 'npz' and 'npz_path'
    npz_key = "npz" if "npz" in paths else ("npz_path" if "npz_path" in paths else None)
    npz_path = paths.get(npz_key) if npz_key else None

    # =========================
    # NPZ fast path
    # =========================
    if npz_path and os.path.exists(npz_path):
        logging.info(f"Loading data from NPZ: {npz_path}")
        data = np.load(npz_path, allow_pickle=False)

        # Allow either your historical names or generic names
        if "snp" in data and target_column in data:
            X = data["snp"].astype(np.float32, copy=False)
            y = data[target_column].astype(np.float32, copy=False)
        elif "X" in data and "y" in data:
            X = data["X"].astype(np.float32, copy=False)
            y = data["y"].astype(np.float32, copy=False)
        else:
            raise ValueError(f"NPZ must contain ('snp','{target_column}') or ('X','y') arrays.")

        if "ids" in data:
            ids = np.asarray(data["ids"]).astype(str)
        else:
            ids = None  # will complain below if GRM is requested

        # ---- Load GRM from RDS (even in NPZ mode), align to ids
        grm_rds = paths.get("grm_rds")
        if grm_rds:
            if pyreadr is None:
                raise ImportError("pyreadr is required to read .rds GRM files. pip install pyreadr")
            logging.info(f"Loading GRM from RDS: {grm_rds}")
            grm_res = pyreadr.read_r(grm_rds)
            # pyreadr returns a dict-like; take the first object
            GRM_df = next(iter(grm_res.values()))
            GRM_df.index = GRM_df.index.astype(str)
            GRM_df.columns = GRM_df.columns.astype(str)

            if ids is None:
                raise ValueError(
                    "NPZ did not contain 'ids' but 'grm_rds' is set. "
                    "Please include 'ids' in the NPZ to align with GRM."
                )

            # Intersect and align to current ids order
            ids_in = np.array([i for i in ids if i in GRM_df.index], dtype=str)
            if len(ids_in) == 0:
                raise ValueError("No overlapping IDs between NPZ 'ids' and GRM index.")
            # Reindex GRM to those ids in the NPZ order
            GRM_df = GRM_df.loc[ids_in, ids_in]

            # If NPZ ids included non-GRM samples, drop them from X/y/ids (keep GRM order)
            if len(ids_in) != len(ids) or not np.array_equal(ids_in, ids):
                idx_map = {v: k for k, v in enumerate(ids)}
                sel = np.array([idx_map[v] for v in ids_in], dtype=int)
                X = X[sel]
                y = y[sel]
                ids = ids_in

    # =========================
    # Raw feather/CSV path
    # =========================
    else:
        if "snp_feather" not in paths or "phenotype_csv" not in paths:
            raise ValueError(
                "Provide either 'npz' (or 'npz_path') OR both 'snp_feather' and 'phenotype_csv' in paths."
            )

        logging.info(f"Loading SNP data from feather: {paths['snp_feather']}")
        snp = pd.read_feather(paths["snp_feather"])
        if "ringnr" not in snp.columns:
            raise ValueError("SNP feather must contain a 'ringnr' column.")
        snp = snp.set_index("ringnr")
        snp.index = snp.index.astype(str)

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
        if len(ids_common) == 0:
            raise ValueError("No overlapping IDs between SNP feather and phenotype CSV.")
        snp = snp.loc[ids_common]
        pheno = pheno.loc[ids_common]
        ids = ids_common

        # NaN check
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
            GRM_df = next(iter(grm_res.values()))
            GRM_df.index = GRM_df.index.astype(str)
            GRM_df.columns = GRM_df.columns.astype(str)

            # Align GRM to current ids order; drop missing
            ids_in = np.array([i for i in ids if i in GRM_df.index], dtype=str)
            GRM_df = GRM_df.loc[ids_in, ids_in]

            # If GRM dropped some individuals, realign X/y/ids to GRM order
            if len(ids_in) != len(ids) or not np.array_equal(ids_in, ids):
                idx_map = {v: k for k, v in enumerate(ids)}
                sel = np.array([idx_map[v] for v in ids_in], dtype=int)
                X = X[sel]
                y = y[sel]
                ids = ids_in

    # Optional standardization (usually not needed for 0/1/2 SNPs)
    if standardize_features:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    return X, y, ids, GRM_df