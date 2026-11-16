using TOML, LinearAlgebra, Logging, Printf
const ROOT = normpath(joinpath(@__DIR__, ".."))
const LOAD_START = time()
using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium
BLAS.set_num_threads(1)
const LOAD_SECONDS = time() - LOAD_START
const CHECKOUT = realpath(joinpath(ROOT, TOML.parsefile(joinpath(ROOT, "dependency.toml"))["path"]))
realpath(pkgdir(GPEC)) == CHECKOUT || error("Julia loaded GPEC from a different checkout")
const PIN = strip(read(joinpath(ROOT, "GPEC_COMMIT"), String))
strip(read(`git -C $CHECKOUT rev-parse HEAD`, String)) == PIN || error("GPEC checkout differs from the pinned commit")
isempty(strip(read(`git -C $CHECKOUT status --porcelain --untracked-files=no`, String))) || error("Pinned GPEC checkout has tracked modifications")

function write_toml(path, data)
    temp = path * ".tmp"
    open(io -> TOML.print(io, data), temp, "w")
    mv(temp, path; force=true)
end

function run_job(job, output)
    dataset = TOML.parsefile(joinpath(ROOT, "inputs", job["dataset"] * ".toml"))
    point = only(filter(p -> p["id"] == job["point"], dataset["points"]))
    preset = get(job, "preset", "baseline")
    name = job["dataset"] * "_" * job["point"] * "_" * preset
    run = joinpath(output, name)
    isdir(run) && error("Output already exists: $run; use a fresh output directory")
    mkpath(run)
    source = joinpath(ROOT, point["equilibrium"])
    actual_hash = first(split(read(`shasum -a 256 $source`, String)))
    actual_hash == point["sha256"] || error("Equilibrium checksum mismatch: $source")
    cp(source, joinpath(run, "equilibrium.geqdsk"))
    preset_path = joinpath(ROOT, "configs", preset * ".toml")
    deck = TOML.parsefile(preset_path)
    preset_hash = first(split(read(`shasum -a 256 $preset_path`, String)))
    write_toml(joinpath(run, "gpec.toml"), deck)
    row = Dict{String,Any}("dataset" => job["dataset"], "point" => point["id"], "preset" => preset,
        "multiplier" => point["multiplier"], "producer_beta_n" => point["producer_beta_n"], "input_sha256" => actual_hash,
        "gpec_commit" => PIN, "config_sha256" => preset_hash, "run_directory" => name)
    println("START ", name)
    flush(stdout)
    started = time()
    try
        result = open(joinpath(run, "solver.log"), "w") do io
            redirect_stdio(; stdout=io, stderr=io) do
                with_logger(ConsoleLogger(io, Logging.Info)) do
                    GPEC.main([run])
                end
            end
        end
        ffs = result.ffs
        free = ffs.free_boundary
        index = argmin(real.(free.et))
        energy = free.et[index]
        wp, wv = free.wp, free.wv
        anti(a) = norm(a - a') / max(norm(a + a'), eps())
        row["status"] = "completed"
        row["beta_n"] = result.equil.params.betan
        row["q0"] = result.equil.params.q0
        row["delta_w"] = real(energy)
        row["delta_w_imag"] = imag(energy)
        row["imaginary_energy_fraction"] = abs(imag(energy)) / max(abs(free.ep[index]) + abs(free.ev[index]), eps())
        row["plasma_asymmetry"] = anti(wp)
        row["vacuum_asymmetry"] = anti(wv)
        row["internal_crossings"] = ffs.diagnostics === nothing ? -1 : ffs.diagnostics.nzero
        row["psi_start"] = ffs.psilow
        row["psi_end"] = ffs.psilim
        row["radial_nodes"] = length(result.equil.profiles.xs)
        row["poloidal_modes"] = ffs.mpert
        row["screening_pass"] =
            all(isfinite, (row["beta_n"], real(energy), imag(energy), anti(wp), anti(wv))) &&
            anti(wp) < 1e-3 && anti(wv) < 1e-3 && row["imaginary_energy_fraction"] < 1e-3 && row["internal_crossings"] == 0
    catch exception
        exception isa InterruptException && rethrow()
        row["status"] = "failed"
        row["screening_pass"] = false
        row["error"] = sprint(showerror, exception, catch_backtrace())
    end
    log_path = joinpath(run, "solver.log")
    row["warnings"] = isfile(log_path) ? [strip(split(line, "Warning:"; limit=2)[2]) for line in eachline(log_path) if occursin("Warning:", line)] : String[]
    row["elapsed_seconds"] = time() - started
    write_toml(joinpath(run, "result.toml"), row)
    println("DONE ", name, " status=", row["status"], " beta=", get(row, "beta_n", NaN), " dW=", get(row, "delta_w", NaN))
    flush(stdout)
    return row
end

function validation_jobs(rows, datasets)
    jobs = Dict{String,Any}[]
    for name in datasets
        baseline = sort(filter(r -> r["dataset"] == name && r["preset"] == "baseline", rows); by=r -> r["multiplier"])
        for i in eachindex(baseline)
            row = baseline[i]
            get(row, "screening_pass", false) || break
            i == 1 && row["delta_w"] <= 0 && break
            if i > 1 && row["delta_w"] <= 0 < baseline[i-1]["delta_w"]
                for preset in ("fine", "core", "riccati"), point in (baseline[i-1]["point"], row["point"])
                    push!(jobs, Dict("dataset" => name, "point" => point, "preset" => preset))
                end
                break
            end
        end
    end
    return jobs
end

function main()
    2 <= length(ARGS) <= 3 || error("Usage: run_scans.jl OUTPUT (tokamaker|nsfsim|both) [PRESET]")
    ARGS[2] in ("tokamaker", "nsfsim", "both") || error("Unknown dataset: $(ARGS[2])")
    datasets = ARGS[2] == "both" ? ["tokamaker", "nsfsim"] : [ARGS[2]]
    preset = get(ARGS, 3, "baseline")
    preset in ("baseline", "fine", "core", "riccati") || error("Unknown preset: $preset")
    output = abspath(ARGS[1])
    isdir(output) && !isempty(readdir(output)) && error("Output directory is not empty; use a fresh directory")
    mkpath(output)
    write_toml(
        joinpath(output, "runtime.toml"),
        Dict("gpec_commit" => PIN, "gpec_source" => pathof(GPEC), "julia_version" => string(VERSION),
            "load_seconds" => LOAD_SECONDS, "blas_threads" => BLAS.get_num_threads(), "julia_threads" => Threads.nthreads(), "datasets" => datasets,
            "equilibrium_generators_required" => false, "python_bridge_required" => false)
    )
    rows = Dict{String,Any}[]
    jobs = [Dict("dataset" => name, "point" => point["id"], "preset" => preset)
     for name in datasets for point in TOML.parsefile(joinpath(ROOT, "inputs", name * ".toml"))["points"]]
    for job in jobs
        push!(rows, run_job(job, output))
        write_toml(joinpath(output, "results.toml"), Dict("results" => rows))
    end
    if length(ARGS) == 2
        for job in validation_jobs(rows, datasets)
            push!(rows, run_job(job, output))
            write_toml(joinpath(output, "results.toml"), Dict("results" => rows))
        end
    end
    println("COMPLETE ", length(rows), " cases")
    all(r -> r["status"] == "completed", rows) || error("Some calculations failed; inspect results.toml and solver.log files")
end
main()
