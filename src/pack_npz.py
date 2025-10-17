"""
Pack SNP feather and phenotype CSV into a compressed NPZ for faster loading.

Saves arrays under keys:
    - 'snp': SNP matrix (float32)
    - 'body_mass': target vector (float32) for --target
    - 'y_mean': mean phenotype vector (float32) if available (column configurable)
Optionally saves 'ids' if --include-ids is provided.

Usage:
    python -m src.pack_npz --snp Data/gnn/SNP/ALL/snp_export_body_mass_ALL_geno.feather --pheno Data/gnn/adjusted_thr_tarsus.csv --target y_adjusted --out Data/gnn/snp_thr_tarsus_ALL.npz --include-ids
"""
import argparse
import os
import numpy as np
import pandas as pd
import logging


def main():
    ap = argparse.ArgumentParser(description="Pack SNP + phenotype into NPZ")
    ap.add_argument("--snp", required=True, help="Path to SNP feather file (must contain 'ringnr' column)")
    ap.add_argument("--pheno", required=True, help="Path to phenotype CSV (must contain 'ringnr' and target column)")
    ap.add_argument("--target", default="y_adjusted", help="Target column name in phenotype CSV")
    ap.add_argument("--mean-column", default="y_mean", help="Column name for mean phenotype to store as 'y_mean'")
    ap.add_argument("--out", required=True, help="Output NPZ path")
    ap.add_argument("--include-ids", action="store_true", help="Include 'ids' (ringnr) in NPZ as well")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("pack_npz")

    try:
        log.info("Starting pack_npz")
        log.info("SNP file: %s", args.snp)
        log.info("Phenotype file: %s", args.pheno)
        log.info("Target column: %s", args.target)
        log.info("Mean column: %s", args.mean_column)

        # Read SNP and phenotype
        log.info("Reading SNP feather...")
        snp = pd.read_feather(args.snp)
        log.debug("SNP columns: %s", list(snp.columns)[:10])
        if "ringnr" not in snp.columns:
            raise ValueError("SNP feather must contain a 'ringnr' column as first identifier column.")
        snp = snp.set_index("ringnr")
        snp.index = snp.index.astype(str)
        log.info("Loaded SNPs: %d individuals, %d features", snp.shape[0], snp.shape[1])

        log.info("Reading phenotype CSV...")
        pheno = pd.read_csv(args.pheno)
        if "ringnr" not in pheno.columns:
            raise ValueError("Phenotype CSV must contain a 'ringnr' column.")
        if args.target not in pheno.columns:
            raise ValueError(f"Phenotype CSV must contain the target column '{args.target}'.")
        has_mean = args.mean_column in pheno.columns
        if not has_mean:
            log.warning("Mean column '%s' not found in phenotype CSV; 'y_mean' will not be stored.", args.mean_column)
        pheno = pheno.set_index("ringnr")
        pheno.index = pheno.index.astype(str)
        log.info("Loaded phenotype: %d rows, columns=%d", pheno.shape[0], pheno.shape[1])

        # Intersect and align by ringnr (SNP order preserved)
        log.info("Aligning by ringnr (intersection, SNP order preserved)...")
        ids_snp = snp.index.to_numpy()
        mask = np.isin(ids_snp, pheno.index.to_numpy())
        ids_common = ids_snp[mask]
        dropped_snp = snp.shape[0] - ids_common.shape[0]
        dropped_pheno = pheno.shape[0] - ids_common.shape[0]
        snp = snp.loc[ids_common]
        pheno = pheno.loc[ids_common]
        log.info("Aligned: %d common IDs | dropped: SNP=%d, PHENO=%d", ids_common.shape[0], dropped_snp, dropped_pheno)

        # Validate
        if pheno[args.target].isna().any():
            n_nan = int(pheno[args.target].isna().sum())
            raise ValueError(f"Found {n_nan} NaNs in target after alignment. Check input files.")
        if has_mean and pheno[args.mean_column].isna().any():
            n_nan = int(pheno[args.mean_column].isna().sum())
            raise ValueError(f"Found {n_nan} NaNs in '{args.mean_column}' after alignment. Check input files.")

        # Prepare arrays
        X = snp.values.astype(np.float32)
        y = pheno[args.target].values.astype(np.float32)
        y_mean = pheno[args.mean_column].values.astype(np.float32) if has_mean else None
        ids = ids_common.astype(str)
        locality = pheno["locality"].values.astype(str)
        log.info(
            "Prepared arrays: X=%s, y=%s, y_mean=%s, ids=%s, locality=%s",
            X.shape,
            y.shape,
            y_mean.shape if y_mean is not None else None,
            "yes" if args.include_ids else "no",
            locality.shape,
        )

        assert X.shape[0] == y.shape[0] == len(ids), \
            f"Mismatch: X={X.shape[0]}, y={y.shape[0]}, ids={len(ids)}"
        if has_mean:
            assert y_mean.shape[0] == X.shape[0], \
                f"Mismatch: y_mean={y_mean.shape[0]} vs X={X.shape[0]}"
        if "locality" in pheno.columns:
            assert locality.shape[0] == X.shape[0], \
                f"Mismatch: locality={locality.shape[0]} vs X={X.shape[0]}"

        # Save
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        log.info("Saving NPZ to: %s", args.out)
        if args.include_ids:
            if y_mean is not None:
                np.savez_compressed(args.out, snp=X, y_adjusted=y, y_mean=y_mean, ids=ids, locality=locality)
            else:
                np.savez_compressed(args.out, snp=X, y_adjusted=y, ids=ids, locality=locality)
        else:
            if y_mean is not None:
                np.savez_compressed(args.out, snp=X, y_adjusted=y, y_mean=y_mean, locality=locality)
            else:
                np.savez_compressed(args.out, snp=X, y_adjusted=y, locality=locality)
        try:
            size_mb = os.path.getsize(args.out) / (1024 * 1024)
            log.info("Saved. File size: %.2f MB", size_mb)
        except OSError:
            log.info("Saved.")

    except Exception as e:
        log.exception("Failed to pack NPZ: %s", e)
        raise


if __name__ == "__main__":
    main()
