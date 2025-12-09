#!/bin/sh
#SBATCH --account=share-ie-imf
#SBATCH --job-name=gcn_w
#SBATCH --time=3-00:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1              # Setting the number of GPUs to 1
#SBATCH --constraint="a100"        # Requesting A100 GPUs
#SBATCH --mem=64G                 # Asking for 64GB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8         # Request 8 CPU cores per task
#SBATCH --ntasks-per-node=1                  # Number of tasks (keep 1 if training a single model)
#SBATCH --output=logs/gcn_w.out
#SBATCH --error=logs/gcn_w.err

#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Threads per (single) trial
export THREADS_PER_TRIAL=4

# Let MKL/OMP use more threads; keep OpenBLAS at 1 to avoid the OpenMP-loop warning
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=$THREADS_PER_TRIAL
export MKL_NUM_THREADS=$THREADS_PER_TRIAL
export NUMEXPR_NUM_THREADS=$THREADS_PER_TRIAL

# Running your python file
module load Anaconda3/2024.02-1
conda activate project_thesis
python -m src.nested_cv --config configs/config_nested_within_gcn_wing.json