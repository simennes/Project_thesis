"""Cross-validation utilities for nested CV pipeline."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from sklearn.model_selection import KFold


# ------------------------- Island naming --------------------------------
# Map known island numeric codes to human-readable names.
ISLAND_ID_TO_NAME: Dict[int, str] = {
    20: "Nesøy",
    22: "Myken",
    23: "Træna",
    24: "Selvær",
    26: "Gjerøy",
    27: "Hestmannøy",
    28: "Indre Kvarøy",
    33: "Onøy og Lurøy",
    34: "Lovund",
    35: "Sleneset",
    38: "Aldra",
    # Southern islands grouped/renamed
    60: "Southern 1",
    61: "Southern 2",
    63: "Southern 3",
    67: "Southern 4",
    68: "Southern 5",
}


def island_label(isl_id: Optional[int], code_to_label: Optional[Dict[int, str]]) -> str:
    """Convert island ID to human-readable name."""
    if isl_id is None:
        return "None"
    try:
        isl_int = int(isl_id)
    except Exception:
        return str(isl_id)
    if isl_int in ISLAND_ID_TO_NAME:
        return ISLAND_ID_TO_NAME[isl_int]
    if code_to_label and isl_int in code_to_label:
        return ISLAND_ID_TO_NAME[int(code_to_label[isl_int])]
    return str(isl_int)


# ---------------------------- CV split generators --------------------------------

def make_outer_splits(strategy: str, locality: np.ndarray, n_splits: int, shuffle: bool, 
                      random_state: int, n: int, predefined_folds: Optional[list] = None, 
                      ids: Optional[np.ndarray] = None):
    """Generate outer CV splits based on strategy.
    
    Parameters
    ----------
    strategy : str
        CV strategy: "leave_island_out" or "kfold"
    locality : np.ndarray
        Island/location codes for each sample
    n_splits : int
        Number of folds for k-fold CV
    shuffle : bool
        Whether to shuffle for k-fold
    random_state : int
        Random seed
    n : int
        Total number of samples
    predefined_folds : list, optional
        Predefined test sets for each fold
    ids : np.ndarray, optional
        Sample IDs (required for predefined folds)
        
    Yields
    ------
    tr_idx : np.ndarray
        Training indices
    te_idx : np.ndarray
        Test indices  
    island_id : int or None
        Island ID for leave-island-out, None for k-fold
    """
    if strategy == "leave_island_out":
        # One outer fold per unique island code
        uniq = np.unique(locality)
        for isl in uniq:
            te = np.where(locality == isl)[0]
            tr = np.where(locality != isl)[0]
            yield (tr, te, int(isl))
    elif strategy == "kfold" and predefined_folds is not None:
        # Use predefined folds from JSON file
        if ids is None:
            raise ValueError("IDs must be provided when using predefined folds")
        
        # Convert ids to strings for matching (JSON IDs are strings)
        id_to_idx = {str(id_val): idx for idx, id_val in enumerate(ids)}
        
        for fold_idx, test_ids in enumerate(predefined_folds):
            # Map test IDs to indices
            te = []
            for test_id in test_ids:
                if str(test_id) in id_to_idx:
                    te.append(id_to_idx[str(test_id)])
            
            te = np.array(te, dtype=int)
            tr = np.array([i for i in range(n) if i not in te], dtype=int)
            
            if len(te) == 0:
                continue
                
            yield (tr, te, None)
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for tr, te in kf.split(np.arange(n)):
            yield (tr, te, None)


def make_inner_splits(idx_train: np.ndarray, n_splits: int, shuffle: bool, random_state: int):
    """Generate inner k-fold splits within the training set.
    
    Parameters
    ----------
    idx_train : np.ndarray
        Training indices from outer split
    n_splits : int
        Number of inner folds
    shuffle : bool
        Whether to shuffle
    random_state : int
        Random seed
        
    Yields
    ------
    train_idx : np.ndarray
        Inner training indices
    val_idx : np.ndarray
        Inner validation indices
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for tr, va in kf.split(idx_train):
        yield (idx_train[tr], idx_train[va])


def make_inner_loio_splits(locality: np.ndarray, idx_outer_train: np.ndarray):
    """Inner LOIO (Leave-One-Island-Out) within the outer-train set.
    
    Parameters
    ----------
    locality : np.ndarray
        Island/location codes for each sample
    idx_outer_train : np.ndarray
        Outer training indices
        
    Returns
    -------
    list of tuples
        (train_idx, val_idx, island_id) for each inner fold
    """
    loc_tr = locality[idx_outer_train]
    uniq = np.unique(loc_tr)
    splits = []
    for isl in uniq:
        val_mask = (loc_tr == isl)
        val_idx = idx_outer_train[val_mask]
        train_idx = idx_outer_train[~val_mask]
        splits.append((train_idx, val_idx, int(isl)))
    return splits
