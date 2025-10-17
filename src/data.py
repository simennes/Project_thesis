# src/data.py
import os
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

def _filter_min_count(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    GRM_df: Optional[pd.DataFrame],
    locality: np.ndarray,
    min_count: int,
    y_eval: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame], np.ndarray, Optional[np.ndarray]]:
    """
    Filter samples to ensure each locality has at least min_count samples.
    Returns filtered X, y, ids, GRM_df, locality.
    """
    if locality is None:
        raise ValueError("locality array is required for min_count filtering.")
    unique, counts = np.unique(locality, return_counts=True)
    to_keep = set(unique[counts >= min_count])
    mask = np.isin(locality, list(to_keep))
    Xf = X[mask]
    yf = y[mask]
    idsf = ids[mask]
    if GRM_df is not None:
        GRMf = GRM_df[mask].loc[idsf, idsf]
    else:
        GRMf = None
    locf = locality[mask]
    y_eval_f = y_eval[mask] if y_eval is not None else None
    return Xf, yf, idsf, GRMf, locf, y_eval_f

def load_data(
    paths,
    target_column: str = "y_adjusted",
    standardize_features: bool = False,
    return_locality: bool = False,
    min_count: int = 0,
    *,
    return_eval: bool = False,
    eval_target_column: Optional[str] = None,
):
    """
    Load SNP, phenotype/targets, and optionally GRM; supports fast path via NPZ.
    If return_locality=True, also returns a locality (island) vector aligned to the final ids.

    Returns:
      If return_locality=False:
        X, y, ids, GRM_df
      If return_locality=True:
        X, y, ids, GRM_df, locality_codes, code_to_label
    """
    logger = logging.getLogger(__name__)

    # ---- Outputs
    X: np.ndarray
    y: np.ndarray
    ids = None
    GRM_df = None
    locality_arr = None  # raw labels aligned with ids at each step
    y_eval: Optional[np.ndarray] = None

    # Accept both 'npz' and 'npz_path'
    npz_key = "npz" if "npz" in paths else ("npz_path" if "npz_path" in paths else None)
    npz_path = paths.get(npz_key) if npz_key else None

    # =========================
    # NPZ fast path
    # =========================
    if npz_path and os.path.exists(npz_path):
        logger.info(f"Loading data from NPZ: {npz_path}")
        data = np.load(npz_path, allow_pickle=False)

        # X/y
        if "snp" in data and target_column in data:
            X = data["snp"].astype(np.float32, copy=False)
            y = data[target_column].astype(np.float32, copy=False)
        elif "X" in data and "y" in data:
            X = data["X"].astype(np.float32, copy=False)
            y = data["y"].astype(np.float32, copy=False)
        else:
            raise ValueError(f"NPZ must contain ('snp','{target_column}') or ('X','y').")

        # Optional evaluation target from NPZ
        if return_eval and eval_target_column:
            if eval_target_column in data.files:
                y_eval = data[eval_target_column].astype(np.float32, copy=False)
            elif eval_target_column == "y_mean" and "y_eval_target" in data.files:
                # historical alias
                y_eval = data["y_eval_target"].astype(np.float32, copy=False)
            else:
                logging.getLogger(__name__).warning(
                    f"Eval target '{eval_target_column}' not found in NPZ; will fallback to training target later."
                )

        # ids (needed for GRM & locality alignment)
        if "ids" in data:
            ids = np.asarray(data["ids"]).astype(str)
        else:
            ids = None

        # locality (optional at this stage; align to ids if present)
        if "locality" in data and ids is not None:
            loc_raw = np.asarray(data["locality"]).ravel()
            if len(loc_raw) != len(ids):
                raise ValueError(f"NPZ 'locality' length {len(loc_raw)} != 'ids' length {len(ids)}.")
            # keep as strings for now (will encode to ints at the end)
            locality_arr = loc_raw.astype(str)

        # ---- Load & align GRM if provided
        grm_rds = paths.get("grm_rds")
        if grm_rds:
            try:
                import pyreadr  # type: ignore
            except Exception as e:
                raise ImportError("pyreadr is required to read .rds GRM files. pip install pyreadr") from e
            logger.info(f"Loading GRM from RDS: {grm_rds}")
            grm_res = pyreadr.read_r(grm_rds)
            GRM_df = next(iter(grm_res.values()))
            GRM_df.index = GRM_df.index.astype(str)
            GRM_df.columns = GRM_df.columns.astype(str)

            if ids is None:
                raise ValueError(
                    "NPZ did not contain 'ids' but 'grm_rds' is set. "
                    "Include 'ids' in the NPZ to align with GRM."
                )

            # Intersect/align to NPZ ids (preserving NPZ order)
            ids_in = np.array([i for i in ids if i in GRM_df.index], dtype=str)
            if len(ids_in) == 0:
                raise ValueError("No overlapping IDs between NPZ 'ids' and GRM index.")

            # If GRM drops some samples, realign X/y/(ids, locality_arr, y_eval) to ids_in
            if len(ids_in) != len(ids) or not np.array_equal(ids_in, ids):
                idx_map = {v: k for k, v in enumerate(ids)}
                sel = np.array([idx_map[v] for v in ids_in], dtype=int)
                X = X[sel]
                y = y[sel]
                if locality_arr is not None:
                    locality_arr = locality_arr[sel]
                if y_eval is not None:
                    y_eval = y_eval[sel]
                ids = ids_in

            # Reindex GRM to the final ids order
            GRM_df = GRM_df.loc[ids, ids]

    # =========================
    # Raw feather/CSV path
    # =========================
    else:
        if "snp_feather" not in paths or "phenotype_csv" not in paths:
            raise ValueError(
                "Provide either 'npz' (or 'npz_path') OR both 'snp_feather' and 'phenotype_csv' in paths."
            )

        logger.info(f"Loading SNP data from feather: {paths['snp_feather']}")
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
        # Optional eval target from CSV
        if return_eval and eval_target_column and eval_target_column in pheno.columns:
            pass  # handled after alignment
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
        if return_eval and eval_target_column:
            if eval_target_column in pheno.columns:
                y_eval = pheno[eval_target_column].values.astype(np.float32)
                if np.isnan(y_eval).any():
                    raise ValueError(f"Found NaNs in eval target '{eval_target_column}' after alignment.")
            else:
                logging.getLogger(__name__).warning(
                    f"Eval target '{eval_target_column}' not found in phenotype CSV; will fallback to training target."
                )

        # Locality from phenotype CSV if present
        if "locality" in pheno.columns:
            locality_arr = pheno["locality"].astype(str).to_numpy()

        # Optional GRM
        grm_rds = paths.get("grm_rds")
        if grm_rds:
            try:
                import pyreadr  # type: ignore
            except Exception as e:
                raise ImportError("pyreadr is required to read .rds GRM files. pip install pyreadr") from e
            logger.info(f"Loading GRM from RDS: {grm_rds}")
            grm_res = pyreadr.read_r(grm_rds)
            GRM_df = next(iter(grm_res.values()))
            GRM_df.index = GRM_df.index.astype(str)
            GRM_df.columns = GRM_df.columns.astype(str)

            # Align GRM to current ids; drop missing → apply same selection to X/y/locality/ids/(y_eval)
            ids_in = np.array([i for i in ids if i in GRM_df.index], dtype=str)
            if len(ids_in) == 0:
                raise ValueError("No overlapping IDs between SNP/PHENO and GRM index.")
            if len(ids_in) != len(ids) or not np.array_equal(ids_in, ids):
                idx_map = {v: k for k, v in enumerate(ids)}
                sel = np.array([idx_map[v] for v in ids_in], dtype=int)
                X = X[sel]
                y = y[sel]
                if locality_arr is not None:
                    locality_arr = locality_arr[sel]
                if y_eval is not None:
                    y_eval = y_eval[sel]
                ids = ids_in
            GRM_df = GRM_df.loc[ids, ids]

    # Optional standardization
    if standardize_features:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    # --------------------------
    # Final: locality encoding
    # --------------------------
    code_to_label = None
    locality_codes = None
    if return_locality:
        if min_count > 0:
            if return_eval:
                X, y, ids, GRM_df, locality_arr, y_eval = _filter_min_count(
                    X, y, ids, GRM_df, locality_arr, min_count, y_eval=y_eval
                )
            else:
                X, y, ids, GRM_df, locality_arr, _ = _filter_min_count(
                    X, y, ids, GRM_df, locality_arr, min_count, y_eval=None
                )
        # If not already set (e.g., NPZ without 'locality'), try phenotype CSV now
        if locality_arr is None:
            phen = paths.get("phenotype_csv")
            if phen and os.path.exists(phen):
                df_loc = pd.read_csv(phen, dtype={"ringnr": str})
                if "locality" in df_loc.columns:
                    df_loc = df_loc.drop_duplicates(subset=["ringnr"], keep="first")
                    loc_map = dict(zip(df_loc["ringnr"].astype(str).to_numpy(),
                                       df_loc["locality"].astype(str).to_numpy()))
                    miss = [rid for rid in ids if rid not in loc_map]
                    if miss:
                        raise ValueError(f"'locality' missing for {len(miss)} ids; e.g. {', '.join(miss[:5])}")
                    locality_arr = np.array([loc_map[rid] for rid in ids], dtype=str)
                else:
                    raise ValueError("No 'locality' found: neither NPZ nor phenotype CSV contain it.")
            else:
                raise ValueError("No 'locality' found: neither NPZ nor phenotype CSV contain it.")

        # Encode to integer codes (preserve original labels via mapping)
        # If array looks numeric, coerce to int; otherwise factorize
        try:
            locality_codes = locality_arr.astype(int)
            # Create label mapping (int->str) even if already ints, to be safe when saving
            uniq_labels = np.unique(locality_arr)
            # ensure deterministic order by sorting by their int value
            uniq_sorted = np.array(sorted(uniq_labels, key=lambda s: int(s)))
            label_to_code = {lbl: i for i, lbl in enumerate(uniq_sorted)}
            # remap to consecutive codes 0..K-1 (so splits are clean)
            locality_codes = np.array([label_to_code[lbl] for lbl in locality_arr], dtype=int)
            code_to_label = {int(v): str(k) for k, v in label_to_code.items()}
        except Exception:
            # Non-numeric labels
            labels, codes = np.unique(locality_arr, return_inverse=True)
            locality_codes = codes.astype(int)
            code_to_label = {int(i): str(lbl) for i, lbl in enumerate(labels)}

    # If evaluation target requested but not found, default to training target
    if return_eval and y_eval is None:
        y_eval = y.copy()

    if not return_locality:
        if return_eval:
            return X, y, ids, GRM_df, y_eval
        else:
            return X, y, ids, GRM_df
    else:
        if return_eval:
            return X, y, ids, GRM_df, locality_codes, code_to_label, y_eval
        else:
            return X, y, ids, GRM_df, locality_codes, code_to_label