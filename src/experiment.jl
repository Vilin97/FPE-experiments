using JLD2, TimerOutputs
using Random: seed!

include("utils.jl")
include("sbtm.jl")
include("landau/sbtm.jl")
include("blob.jl")

# TODO change saving to only save trajectories and kwargs, not the full ODESolution.

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

function landau_experiment()
    # ns = [200, 1000, 2000, 4000]
    ns = [1000]
    reset_timer!()
    for n in ns
        println("n = $n")
        seed!(1234)
        xs, ts, A, ρ = landau(n)
        @timeit "n = $n" solution = sbtm(xs, ts, A; ρ₀ = x->ρ(x,0.))
        JLD2.save("$landau_experiment/sbtm_n_$(n).jld2", "solution", solution)
    end
    print_timer()
end

landau_experiment()