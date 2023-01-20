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
    epochs = initialize_s!(s, ρ₀, xs; verbose = verbose, kwargs...)
    verbose > 0 && println("Took $epochs epochs to initialize. Initial loss: $(loss(s,xs))")
    return s
end

# TODO split sbtm and jhu into separate functions so that I can run only one of them if needed
function moving_trap_experiment(N=50, num_samples=100, num_timestamps=200; folder = "data")
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    println("Done with initial setup.")

    ε = 128.
    (solution, jhu_trajectories), t = @timed jhu(xs, Δts, b, D, ε)
    println("Done with jhu. Took $t seconds.")

    s, t = @timed initialize_s(ρ₀, xs, 100, 1, verbose = 1)
    epochs = 25
    println("Done with NN initialization. Took $t seconds.")
    (sbtm_trajectories, losses, s_values), t = @timed sbtm(xs, Δts, b, D, s; epochs = epochs, record_losses = true, verbose = 0)
    println("Done with sbtm. Took $t seconds.")

    JLD2.save("$(folder)/moving_trap_data_$seed.jld2", 
        "sbtm_trajectories", sbtm_trajectories, 
        "losses", losses, 
        "s_values", s_values, 
        "jhu_trajectories", jhu_trajectories,
        "epsilon", ε,
        "seed", seed)

    println("Done with saving")
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

# moving_trap_jhu_epsilon_experiment()
moving_trap_experiment(folder = "data")