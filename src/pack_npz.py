"""
Pack SNP feather and phenotype CSV into a compressed NPZ for faster loading.

Saves arrays under keys:
  - 'snp': SNP matrix (float32)
  - 'body_mass': target vector (float32)
Optionally saves 'ids' if --include-ids is provided.

Usage:
  python -m src.pack_npz --snp Data/gnn/SNP/ALL/snp_export_body_mass_ALL_geno.feather --pheno Data/gnn/adjusted_body_mass.csv --target y_adjusted --out Data/gnn/snp_pheno_ALL.npz
"""
import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Pack SNP + phenotype into NPZ")
    ap.add_argument("--snp", required=True, help="Path to SNP feather file (must contain 'ringnr' column)")
    ap.add_argument("--pheno", required=True, help="Path to phenotype CSV (must contain 'ringnr' and target column)")
    ap.add_argument("--target", default="y_adjusted", help="Target column name in phenotype CSV")
    ap.add_argument("--out", required=True, help="Output NPZ path")
    ap.add_argument("--include-ids", action="store_true", help="Include 'ids' (ringnr) in NPZ as well")
    args = ap.parse_args()

    # Read SNP and phenotype
    snp = pd.read_feather(args.snp)
    if "ringnr" not in snp.columns:
        raise ValueError("SNP feather must contain a 'ringnr' column as first identifier column.")
    snp = snp.set_index("ringnr")
    snp.index = snp.index.astype(str)

    pheno = pd.read_csv(args.pheno)
    if "ringnr" not in pheno.columns:
        raise ValueError("Phenotype CSV must contain a 'ringnr' column.")
    if args.target not in pheno.columns:
        raise ValueError(f"Phenotype CSV must contain the target column '{args.target}'.")
    pheno = pheno.set_index("ringnr")
    pheno.index = pheno.index.astype(str)

    # Intersect and align by ringnr (SNP order preserved)
    ids_snp = snp.index.to_numpy()
    mask = np.isin(ids_snp, pheno.index.to_numpy())
    ids_common = ids_snp[mask]
    snp = snp.loc[ids_common]
    pheno = pheno.loc[ids_common]

    if pheno[args.target].isna().any():
        raise ValueError("Found NaNs in target after alignment. Check input files.")

    X = snp.values.astype(np.float32)
    y = pheno[args.target].values.astype(np.float32)
    ids = ids_common.astype(str)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.include_ids:
        np.savez_compressed(args.out, snp=X, body_mass=y, ids=ids)
    else:
        np.savez_compressed(args.out, snp=X, body_mass=y)
    print(f"Saved NPZ to {args.out}: snp={X.shape}, body_mass={y.shape}, ids={'yes' if args.include_ids else 'no'}")


if __name__ == "__main__":
    main()
