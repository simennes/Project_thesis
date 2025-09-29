#!/bin/sh
#SBATCH --account=share-ie-imf
#SBATCH --job-name=tune_params
#SBATCH --time=0-01:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1              # Setting the number of GPUs to 1
#SBATCH --constraint="v100|a100"
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16         # Request 16 CPU cores per task
#SBATCH --ntasks-per-node=1                  # Number of tasks (keep 1 if training a single model)
#SBATCH --output=logs/log.txt      # Specifying 'stdout'
#SBATCH --error=logs/log.err       # Specifying 'stderr'

#SBATCH --mail-user=simen.nesland@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2024.02-1
conda activate project_thesis
python -m src.cross_validate --config config_validate.json