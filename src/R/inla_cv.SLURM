#!/bin/bash
#SBATCH --account=share-ie-imf
#SBATCH --job-name=inla_cv
#SBATCH --time=4-00:00:00
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH --error=logs/log_R.err
#SBATCH --output=logs/log_R.out

set -euo pipefail
mkdir -p logs outputs/nested_cv

module purge
module load R/4.4.2-gfbf-2024a
module load R-bundle-CRAN/2024.11-foss-2024a
module load CMake/3.29.3-GCCcore-13.3.0

# Use the same user library as during install
export R_LIBS_USER="$HOME/.Rlibs-gfbf-2024a"

# Never use MKL binary
export INLA_MKL="0"

# Ensure INLA temp dir is local and writable
export INLA_TMPDIR="${SLURM_SUBMIT_DIR}/outputs/inla_tmp"
mkdir -p "$INLA_TMPDIR"

# Help INLA find its external binary if already downloaded to user home
export PATH="$HOME/.inla/bin/linux/64bit:$HOME/.inla/bin:$PATH"

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$SLURM_SUBMIT_DIR"

Rscript --vanilla src/R/run_loio_inla_gblup.R