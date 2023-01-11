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

function moving_trap_experiment()
    N, num_samples, num_timestamps = 50, 100, 200
    seed = N*num_samples*num_timestamps
    seed!(seed)
    t0 = time()
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    ε = Float32(1.24)
    t1 = time()
    println("Done with initial setup. Took $(t1-t0) seconds.")
    jhu_trajectories = jhu(xs, Δts, b, D, ε)
    t2 = time()
    # animation = animate_2d(jhu_trajectories, "jhu", Δts; target = target, plot_every = 1, fps = 5)

    println("Done with jhu. Took $(t2-t1) seconds.")

    s = initialize_s(ρ₀, xs, 100, 1, verbose = 1)
    epochs = 25
    t3 = time()
    println("Done with NN initialization. Took $(t3-t2) seconds.")
    sbtm_trajectories, losses, s_values = sbtm(xs, Δts, b, D, s; epochs = epochs, record_losses = true, verbose = 0)
    t4 = time()
    # animation = animate_2d(trajectories, "sbtm", Δts, samples = 2, fps = 10, target = target)
    # loss_plot = plot_losses(losses)

    println("Done with sbtm. Took $(t4-t3) seconds.")

    JLD2.save("moving_trap_data_$seed.jld2", 
        "sbtm_trajectories", sbtm_trajectories, 
        "losses", losses, 
        "s_values", s_values, 
        "jhu_trajectories", jhu_trajectories,
        "seed", seed)

    println("Done with saving")
end
moving_trap_experiment()