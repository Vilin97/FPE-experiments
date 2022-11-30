# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, Plots
using Distributions: MvNormal, logpdf
using Zygote: gradient, pullback
using Flux.Optimise: Adam
using Flux: params, train!
using Statistics: mean
using Flux.OneHotArrays: onehot

include("utils.jl")

# divergence of vector field s at x
function divergence(f, v)
    _, ∂f = pullback(f, v)
    sum(eachindex(v)) do i
        ∂fᵢ = ∂f(onehot(i, eachindex(v)))
        sum(x -> x[i], ∂fᵢ)
    end
end
# divergence(s, x) = sum(gradient(x -> s(x)[i], x)[1][i] for i in 1:length(x))
loss(s, xs :: Array{T, 3}) where T =  (sum(s(xs).^2) + T(2.0)*divergence(s, xs))/size(xs, 3)

score(ρ, x) = convert(eltype(x), sum(logpdf(ρ, x)))
propagate(x, t, Δt, b, D, s) = x + Δt * b(x, t) + D(x, t)*s(x)

function plot_losses(losses, epochs)
    p = plot(vec(losses), title = "Score approximation", xaxis = "epochs", yaxis = "loss", label = "training loss")
    scatter!(p, 1:epochs:length(vec(losses)), vec(losses)[1:epochs:end], label = "discrete time propagation", marker = true)
end

function custom_train!(s, xs; optimiser = Adam(5. * 10^-3), epochs = 25, record_losses = true)
    θ = params(s)
    losses = zeros(typeof(loss(s, xs)), epochs)
    for epoch in 1:epochs
        record_losses && (losses[epoch] = loss(s, xs))
        @show epoch, losses[epoch]
        grads = gradient(() -> loss(s, xs), θ)
        Flux.update!(optimiser, θ, grads)
    end
    losses
end

function initialize_s(ρ₀, xs, size_hidden, num_hidden; activation = relu, verbose = true, kwargs...)
    d = size(xs, 1)
    s = Chain(
        Dense(d => size_hidden, activation),
        repeat([Dense(size_hidden, size_hidden), activation], num_hidden)...,
        Dense(size_hidden => d))
    epochs = initialize_s!(s, ρ₀, xs; kwargs...)
    verbose && print("Took $epochs epochs to initialize. Initial loss: $(loss(s,xs))")
    return s
end

function initialize_s!(s, ρ₀, xs; optimiser = Adam(10^-4), ε = 10^-4)
    ys = gradient(x -> score(ρ₀, x), xs)[1]
    ys_sum_squares = sum(ys.^2)
    square_error(s) = sum( (s(xs) - ys).^2 / ys_sum_squares )
    epoch = 0
    θ = params(s)
    while square_error(s) > ε
        grads = gradient(() -> square_error(s), θ)
        Flux.update!(optimiser, θ, grads)
        epoch += 1
    end
    epoch
end

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ → Rᵈ
D   : Rᵈ → Rᵈˣᵈ
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm(xs, Δts, b, D, s; kwargs...)
    trajectories = zeros(eltype(xs), size(xs)..., 1+length(Δts)) # trajectories[:, i, j, k] is particle i of sample j at time k
    trajectories[:, :, :, 1] = xs
    losses = sbtm!(trajectories, Δts, b, D, s; kwargs...)
    trajectories, losses
end

function sbtm!(trajectories :: Array{T, 4}, Δts, b, D, s; kwargs...) where T
    t = zero(eltype(Δts))
    losses = zeros(T, epochs, length(Δts))
    for (k, Δt) in enumerate(Δts)
        @show k
        xs_k = trajectories[:, :, :, k]
        losses[:, k] = custom_train!(s, xs_k; kwargs...)
        @show k
        for (j, x) in enumerate(eachslice(xs_k, dims = 3))
            trajectories[:, :, j, k+1] = propagate(x, t, Δt, b, D, s)
        end
        t += Δt
    end
    losses
end

# toy example
args = attractive_origin()
trajectories, losses = sbtm(args...)
plt = plot_losses(losses, 25)
animation = animate_2d(trajectories, samples = 1:9)

# first example from paper
xs, Δts, b, D, ρ₀, target = moving_trap()
s = initialize_s(ρ₀, xs, 32, 1)
epochs = 10
trajectories, losses = sbtm(xs, Δts, b, D, s; optimiser = Adam(10^-3), epochs = epochs, record_losses = true)
animation = animate_2d(trajectories, "sbtm", Δts, samples = 2, fps = 10, target = target)
plot_losses(losses, epochs)

# TODO do not redefine the anonymous functions every time in custom_train!, divergence, etc
# TODO figure out loss blowing up in the stbm
# TODO speed things up somehow...