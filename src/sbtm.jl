# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, Plots
using Distributions: MvNormal, logpdf
using Zygote: gradient, pullback, withgradient
using Flux.Optimise: Adam
using Flux: params, train!
using Statistics: mean
using Flux.OneHotArrays: onehot

include("utils.jl")

# divergence of vector field s at x
function divergence(f, v)
    _, ∂f = pullback(f, v)
    id = I(length(v))
    sum(eachindex(v)) do i
        ∂f( @view id[i,:] )[1][i]
    end
end
loss(s, xs :: AbstractArray{T, 3}) where T =  (sum(x -> x^2, s(xs)) + T(2.0)*divergence(s, xs))/size(xs, 3)

score(ρ, x) = convert(eltype(x), sum(logpdf(ρ, x)))
propagate(x, t, Δt, b, D, s) = x + Δt * (b(x, t) + D(x, t)*s(x))

function plot_losses(losses, epochs)
    p = plot(vec(losses), title = "Score approximation", xaxis = "epochs", yaxis = "loss", label = "training loss")
    scatter!(p, 1:epochs:length(vec(losses)), vec(losses)[1:epochs:end], label = "discrete time propagation", marker = true)
end

function custom_train!(s, losses, xs; optimiser = Adam(5. * 10^-3), epochs = 25, record_losses = true, verbose = true)
    θ = params(s)
    for epoch in 1:epochs
        loss_value, grads = withgradient(() -> loss(s, xs), θ)
        Flux.update!(optimiser, θ, grads)
        record_losses && (losses[epoch] = loss_value)
        verbose && @show epoch, losses[epoch]
    end
    losses
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

function sbtm!(trajectories :: Array{T, 4}, Δts, b, D, s; verbose = true, kwargs...) where T
    t = zero(eltype(Δts))
    losses = zeros(T, epochs, length(Δts))
    for (k, Δt) in enumerate(Δts)
        verbose && @show k
        xs_k = @view trajectories[:, :, :, k]
        losses[:, k] = custom_train!(s, (@view losses[:, k]), xs_k; verbose = verbose, kwargs...)
        trajectories[:, :, :, k+1] = propagate(xs_k, t, Δt, b, D, s)
        t += Δt
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

# first example from paper
xs, Δts, b, D, ρ₀, target = moving_trap()
s = initialize_s(ρ₀, xs, 32, 1)
epochs = 10
trajectories, losses = sbtm(xs, Δts, b, D, s; optimiser = Adam(10^-3), epochs = epochs, record_losses = true, verbose = true)
animation = animate_2d(trajectories, "sbtm", Δts, samples = 2, fps = 10, target = target)
plot_losses(losses, epochs)

# TODO speed things up somehow. Profile sbtm

@time sbtm(xs, Δts, b, D, s; optimiser = Adam(10^-3), epochs = epochs, record_losses = true)
@time sbtm(xs, Δts, b, D, s; optimiser = Adam(10^-3), epochs = epochs, record_losses = true, verbose = false)
@time sbtm(xs, Δts, b, D, s; optimiser = Adam(10^-3), epochs = epochs, record_losses = false, verbose = false)
