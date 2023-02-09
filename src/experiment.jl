using JLD2
using Random: seed!

include("utils.jl")
include("sbtm.jl")
include("blob.jl")

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

function moving_trap_experiment_sbtm_old_new(N, num_samples, num_timestamps; folder = "data")
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    println("Done with initial setup for sbtm.")

    s = initialize_s(ρ₀, xs, 100, 1)
    # seed!(seed)
    # (old_sbtm_trajectories, _), t = @timed sbtm(xs, Δts, b, D; ρ₀ = ρ₀, s=s)
    # println("Done with old sbtm. Took $t seconds.")
    
    seed!(seed)
    (new_sbtm_trajectories, _), t = @timed sbtm(xs, Δts, b, D; ρ₀ = ρ₀, s=s)
    println("Done with new sbtm. Took $t seconds.")

    # JLD2.save("$(folder)/moving_trap_sbtm_$seed.jld2", 
    #     "trajectories", old_sbtm_trajectories, 
    #     "seed", seed,
    #     "N", N,
    #     "num_samples", num_samples,
    #     "num_timestamps", num_timestamps)

    JLD2.save("$(folder)/moving_trap_sbtm_new_$seed.jld2", 
        "trajectories", new_sbtm_trajectories, 
        "seed", seed,
        "N", N,
        "num_samples", num_samples,
        "num_timestamps", num_timestamps)

    println("Done with saving for sbtm")
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

# N=50
# num_samples=100
# num_timestamps=200
# moving_trap_experiment_sbtm_old_new(N, num_samples, num_timestamps, folder = "old_new_sbtm")

num_samples=100
num_timestamps=300
# attractive_origin_experiment_combined(num_samples, num_timestamps, folder = "data")
attractive_origin_experiment(num_samples, num_timestamps, blob, "blob"; folder = "data", ε = 1/π)