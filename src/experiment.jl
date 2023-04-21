using JLD2, TimerOutputs
using Random: seed!

include("utils.jl")
include("sbtm.jl")
include("landau/sbtm.jl")
include("blob.jl")
include("landau/blob.jl")

# first example from paper
function moving_trap_experiment_combined(N, num_samples, num_timestamps; folder = "data")
    moving_trap_experiment(N, num_samples, num_timestamps, sbtm, "sbtm"; folder = folder)
    moving_trap_experiment(N, num_samples, num_timestamps, blob, "blob"; folder = folder)
end

function moving_trap_experiment(N, num_samples, num_timestamps, method, method_name; folder = "data", kwargs...)
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    println("Done with initial setup for $method_name.")

    (trajectories, _), t = @timed method(xs, Δts, b, D; ρ₀ = ρ₀, kwargs...)
    println("Done with $method_name. Took $t seconds.")

    JLD2.save("$(folder)/moving_trap_$(method_name)_$seed.jld2", 
        "trajectories", trajectories,
        "kwargs", kwargs,
        "seed", seed,
        "N", N,
        "num_samples", num_samples,
        "num_timestamps", num_timestamps)

    println("Done with saving for $method_name")
end

function moving_trap_blob_epsilon_experiment(N=50, num_samples=100, num_timestamps=200)
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    epsilons = 0.1:0.02:0.2
    for ε in epsilons
        t1 = time()
        solution, trajectories = blob(xs, Δts, b, D, ε)
        t2 = time()
        println("Done with ε=$(round(ε, digits = 2)). Took $(t2-t1) seconds.")

        JLD2.save("blob_eps_experiment//blob_epsilon_experiment_eps_$(round(ε, digits = 2)).jld2", 
        "blob_trajectories", trajectories,
        "epsilon", ε,
        "seed", seed)
    end

    println("Done with saving")
end

function attractive_origin_experiment_combined(num_samples, num_timestamps; folder = "data")
    attractive_origin_experiment(num_samples, num_timestamps, sbtm, "sbtm"; folder = folder)
    attractive_origin_experiment(num_samples, num_timestamps, blob, "blob"; folder = folder, ε = 0.14)
end

function attractive_origin_experiment(num_samples, num_timestamps, method, method_name; folder = "data", kwargs...)
    seed = num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, ρ = attractive_origin(num_samples, num_timestamps)

    println("Done with initial setup for $method_name.")

    (trajectories, solution), t = @timed method(xs, Δts, b, D; ρ₀ = ρ₀, kwargs...)
    println("Done with $method_name. Took $t seconds.")

    JLD2.save("$(folder)/attractive_origin_$(method_name)_$seed.jld2", 
        "trajectories", trajectories,
        "solution", solution,
        "kwargs", kwargs,
        "seed", seed,
        "num_samples", num_samples,
        "num_timestamps", num_timestamps)

    println("Done with saving for $method_name")
end

"""
Solve pure dimension-d diffusion problems with n particles for different d and n. Save solutions.

Usage: do_experiment([1,2,3,5,10], pure_diffusion, "pure_diffusion", methods = [blob], method_names = ["blob"])
"""
function do_experiment(ds, experiment, experiment_name; methods = [sbtm, blob], method_names = ["sbtm", "blob"], epsilon_choice = epsilon)
    # ns = [50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 4000]
    # ns = [200, 1000, 2000, 4000]
    ns = [200, 1000]
    reset_timer!()
    for d in ds
        println("d = $d")
        @timeit "d = $d" for n in ns
            ε = epsilon_choice(d, n)
            println("  n = $n")
            seed!(1234)
            xs, ts, b, D, ρ₀, ρ = experiment(d, n)
            @timeit "n = $n" for (method, method_name) in zip(methods, method_names)
                @timeit method_name solution = method(xs, ts, b, D; ρ₀ = ρ₀, ε = ε)
                JLD2.save("$(experiment_name)_experiment/$(method_name)_d_$(d)_n_$(n)_eps_$ε.jld2", "solution", solution,
                    "epsilon", ε)
            end
        end
    end
    print_timer()
end

# TODO: train the NN to approximate the initial score once, on many particles. Save it. Then use it for all the other experiments.
function landau_sbtm_experiment(;num_runs = 5, verbose = 1)
    ns = [50, 100, 200, 400, 500, 1000, 2000, 4000, 10_000, 20_000]
    # ns = [50]
    start_time = 6
    pre_trained_s = load("models/landau_model_n_4000_start_6.jld2", "s")
    K(t) = 1 - exp(-(t+start_time)/6)
    reset_timer!()
    for n in ns
        xs, ts, ρ = landau(n, start_time)
        saveat = ts[[1, (length(ts)+1)÷2, end]]
        ρ₀(x) = ρ(x,K(0))
        combined_solution = [zeros(size(xs, 1), size(xs, 2)*num_runs) for _ in saveat]
        println("n = $n")
        combined_s_values = zeros(eltype(xs), size(xs)..., length(ts), num_runs)
        @timeit "n = $n" for run in 1:num_runs
            seed!(run)
            xs, ts, ρ = landau(n, start_time)
            s = deepcopy(pre_trained_s)
            @timeit "initialize NN" initialize_s!(s, ρ₀, xs, loss_tolerance = 1e-4, verbose = verbose, max_iter = 10^4)
            solution, s_values, losses = sbtm_landau(xs, ts; s = s, verbose = verbose, loss_tolerance = 1e-4, saveat = saveat, record_s_values = true)
            for i in 1:length(saveat)
                combined_solution[i][:, (run-1)*size(xs, 2)+1:run*size(xs, 2)] .= solution.u[i]
            end
            combined_s_values[:, :, :, run] .= s_values
        end
        JLD2.save("landau_experiment/sbtm_n_$(n)_runs_$(num_runs)_start_$(start_time).jld2", 
        "solution", combined_solution,
        "s_values", combined_s_values,
        "saveat", saveat,
        "n", n,
        "num_runs", num_runs,
        "start_time", start_time,
        "timer", TimerOutputs.get_defaulttimer())
    end
    print_timer()
end
landau_sbtm_experiment(num_runs=5)

function landau_blob_experiment(;num_runs = 5, verbose = 1)
    ns = [50, 100, 200, 400, 500, 1000, 2000, 4000, 10_000, 20_000]
    # ns = [50]
    start_time = 6
    reset_timer!()
    for n in ns
        ε = 0.05
        xs, ts, ρ = landau(n, start_time)
        saveat = ts[[1, (length(ts)+1)÷2, end]]
        combined_solution = [zeros(size(xs, 1), size(xs, 2)*num_runs) for _ in saveat]
        println("n = $n")
        for run in 1:num_runs
            seed!(run)
            xs, ts, ρ = landau(n, start_time)
            @timeit "n = $n" solution = blob_landau(xs, ts; ε = ε, saveat = saveat, verbose = verbose)
            for i in 1:length(saveat)
                combined_solution[i][:, (run-1)*size(xs, 2)+1:run*size(xs, 2)] .= solution.u[i]
            end
        end
        JLD2.save("landau_experiment/blob_n_$(n)_runs_$(num_runs)_start_$(start_time).jld2", 
        "solution", combined_solution,
        "saveat", saveat,
        "n", n,
        "num_runs", num_runs,
        "start_time", start_time,
        "timer", TimerOutputs.get_defaulttimer())
    end
    print_timer()
end
landau_blob_experiment(num_runs=5)

# sol1=load("landau_experiment/sbtm_n_10000_runs_5.jld2", "solution")
# sol2=load("landau_experiment/blob_n_10000_runs_5.jld2", "solution")