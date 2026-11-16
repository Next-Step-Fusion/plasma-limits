#!/usr/bin/env bash
set -euo pipefail
workflow_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workflow_root"
julia --project=environment -e '
    using Pkg, TOML
    dependency_path = TOML.parsefile("dependency.toml")["path"]
    expected_commit = strip(read("GPEC_COMMIT", String))
    actual_commit = strip(read(`git -C $dependency_path rev-parse HEAD`, String))
    actual_commit == expected_commit || error("GPEC checkout does not match GPEC_COMMIT")
    Pkg.develop(path=dependency_path)
    Pkg.instantiate()
    Pkg.precompile()
'
