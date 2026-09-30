#!/bin/bash
# Submit one scan_SCfp_ESCAPE.slurm job per parameter combination.
# Job name and .out/.err files match the output_dir built in scan_SCfp_ESCAPE.py.
set -euo pipefail
cd "$(dirname "$0")"

sc_models=('1D' '3D')
sc_implementations=('firedrake' 'scipy')
ne_grad_bc_locs=('inner' 'outer')

for sc_model in "${sc_models[@]}"; do
  for sc_implementation in "${sc_implementations[@]}"; do
    for ne_grad_bc_loc in "${ne_grad_bc_locs[@]}"; do
      # Must match output_dir in scan_SCfp_ESCAPE.py
      name="DIIIDSnyder_ESCAPE_${sc_model}${sc_implementation}${ne_grad_bc_loc}"
      sbatch \
        --job-name="${name}" \
        --output="${name}_%j.out" \
        --error="${name}_%j.err" \
        --export=ALL,SC_MODEL="${sc_model}",SC_IMPLEMENTATION="${sc_implementation}",NE_GRAD_BC_LOC="${ne_grad_bc_loc}" \
        scan_SCfp_ESCAPE.slurm
    done
  done
done
