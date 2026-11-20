#!/usr/bin/env bash
set -euo pipefail
workflow_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workflow_root"
output_dir="${1:-outputs/example}"
python3 -c 'import tomllib, matplotlib'
julia --project=environment -t 2 scripts/run_scans.jl "$output_dir" both
python3 scripts/summarize.py "$output_dir" --strict
python3 scripts/plot_scans.py "$output_dir"
