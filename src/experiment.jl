using JLD2
using Random: seed!

include("utils.jl")
include("sbtm.jl")
include("jhu.jl")

# first example from paper

function initialize_s(ρ₀, xs, size_hidden, num_hidden; activation = relu, verbose = 1, kwargs...)
    d_bar, N, n = size(xs)
    d = d_bar*N
    s = Chain(
        # xs -> reshape(xs, d, n),
        Dense(d_bar => size_hidden, activation),
        repeat([Dense(size_hidden, size_hidden), activation], num_hidden-1)...,
        Dense(size_hidden => d_bar)
        # xs -> reshape(xs, d_bar, N, n)
        )
    epochs,t = @timed initialize_s!(s, ρ₀, xs; verbose = verbose, kwargs...)
    println("Done with NN initialization. Took $epochs epochs and $t seconds. Initial loss: $(loss(s,xs))")
    return s
end


function moving_trap_experiment(N, num_samples, num_timestamps; folder = "data")
    moving_trap_experiment_jhu(N, num_samples, num_timestamps; folder = folder)
    moving_trap_experiment_sbtm(N, num_samples, num_timestamps; folder = folder)
end

function moving_trap_experiment_jhu(N, num_samples, num_timestamps; folder = "data")
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    println("Done with initial setup for jhu.")

    ε = 0.14
    (jhu_trajectories, _), t = @timed jhu(xs, Δts, b, D, ε)
    println("Done with jhu. Took $t seconds.")

    JLD2.save("$(folder)/moving_trap_jhu_$seed.jld2", 
        "trajectories", jhu_trajectories,
        "epsilon", ε,
        "seed", seed,
        "N", N,
        "num_samples", num_samples,
        "num_timestamps", num_timestamps)

    println("Done with saving for jhu")
end

function moving_trap_experiment_sbtm(N, num_samples, num_timestamps; folder = "data")
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    println("Done with initial setup for sbtm.")

    s = initialize_s(ρ₀, xs, 100, 1)
    (sbtm_trajectories, losses, s_values), t = @timed sbtm(xs, Δts, b, D, s)
    println("Done with sbtm. Took $t seconds.")

    JLD2.save("$(folder)/moving_trap_sbtm_$seed.jld2", 
        "trajectories", sbtm_trajectories, 
        "losses", losses, 
        "s_values", s_values, 
        "seed", seed,
        "N", N,
        "num_samples", num_samples,
        "num_timestamps", num_timestamps)

    println("Done with saving for sbtm")
end

function moving_trap_jhu_epsilon_experiment(N=50, num_samples=100, num_timestamps=200)
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    epsilons = 0.1:0.02:0.2
    for ε in epsilons
        t1 = time()
        solution, trajectories = jhu(xs, Δts, b, D, ε)
        t2 = time()
        println("Done with ε=$(round(ε, digits = 2)). Took $(t2-t1) seconds.")

        JLD2.save("jhu_eps_experiment//jhu_epsilon_experiment_eps_$(round(ε, digits = 2)).jld2", 
        "jhu_trajectories", trajectories,
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
    seed!(seed)
    (new_sbtm_trajectories, losses, s_values, _), t = @timed sbtm_solve1(xs, Δts, b, D, s)
    println("Done with new sbtm. Took $t seconds.")

    seed!(seed)
    (old_sbtm_trajectories, losses, s_values), t = @timed sbtm(xs, Δts, b, D, s)
    println("Done with old sbtm. Took $t seconds.")

    JLD2.save("$(folder)/moving_trap_sbtm_$seed.jld2", 
        "trajectories", old_sbtm_trajectories, 
        "losses", losses, 
        "s_values", s_values, 
        "seed", seed,
        "N", N,
        "num_samples", num_samples,
        "num_timestamps", num_timestamps)

    JLD2.save("$(folder)/moving_trap_sbtm_new_$seed.jld2", 
        "trajectories", new_sbtm_trajectories, 
        "losses", losses, 
        "s_values", s_values, 
        "seed", seed,
        "N", N,
        "num_samples", num_samples,
        "num_timestamps", num_timestamps)

    println("Done with saving for sbtm")
end

N=50
num_samples=100
num_timestamps=200
moving_trap_experiment_sbtm_old_new(N, num_samples, num_timestamps, folder = "old_new_sbtm")