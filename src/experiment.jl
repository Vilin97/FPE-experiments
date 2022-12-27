using JLD2
using Random: seed!

include("utils.jl")
include("sbtm.jl")
include("jhu.jl")

# first example from paper

function potential(xs, u1, u2)
    d_bar, N, n = size(xs)
    term1 = reshape(sum(u1(xs), dims = 2), n) # first term of the potential in eqn (13)
    # TODO write the second term
end

function initialize_s(ρ₀, xs, size_hidden, num_hidden; activation = relu, verbose = 1, kwargs...)
    d_bar, N, n = size(xs)
    s = Chain(
        xs -> reshape(xs, d_bar*N,n)
        Dense(d => size_hidden, activation),
        repeat([Dense(size_hidden, size_hidden), activation], num_hidden-1)...,
        Dense(size_hidden => d),
        xs -> reshape(xs, d_bar, N, n))
    epochs = initialize_s!(s, ρ₀, xs; kwargs...)
    verbose > 0 && println("Took $epochs epochs to initialize. Initial loss: $(loss(s,xs))")
    return s
end

function moving_trap_experiment()
    d_bar, N, n = 5, 10, 20
    seed = d_bar*N*n
    seed!(seed)
    t0 = time()
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(d_bar, N, n)

    ε = Float32(1/π)
    t1 = time()
    println("Done with initial setup. Took $(t1-t0) seconds.")
    jhu_trajectories = jhu(xs, Δts, b, D, ε)
    t2 = time()
    # animation = animate_2d(jhu_trajectories, "jhu", Δts; target = target, plot_every = 1, fps = 5)

    println("Done with jhu. Took $(t2-t1) seconds.")

    s = initialize_s(ρ₀, xs, 100, 1)
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