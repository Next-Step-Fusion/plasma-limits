#!/bin/bash
#SBATCH --job-name=DIII-D_SCfp_ESCAPE_3Dfiredrakeinner
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=DIII-D_scan%j.out
#SBATCH --error=DIII-D_scan%j.err
#SBATCH --chdir=/mnt/homes_global/jal2351/software/ESCAPE/examples/device_prediction/DIII-D

set -euo pipefail

# Firedrake venv (also sets PETSc env vars and MPLBACKEND=Agg)
source /mnt/codes/jal2351/venv/bin/activate

# Keep threading to a single CPU for this serial run
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python scan_SCfp_ESCAPE.py

echo "Job finished at $(date)"
