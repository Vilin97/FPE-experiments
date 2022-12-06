# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, Plots
using Distributions: MvNormal, logpdf
using Zygote: gradient, pullback, withgradient
using Flux.Optimise: Adam
using Flux: params, train!
using Statistics: mean
using Flux.OneHotArrays: onehot

function divergence1(f, v)
    _, ∂f = pullback(f, v)
    id = I(length(v)) |> gpu
    sum(eachindex(v)) do i
        ∂f( @view id[i,:] )[1][i]
    end
end

divergence3(f,v) = sum(gradient(v -> f(v)[i], v)[1][i] for i in eachindex(v))

# divergence of vector field s at x
function divergence(f, v)
    _, ∂f = pullback(f, v)
    sum(eachindex(v)) do i
        ∂fᵢ = ∂f(onehot(i, eachindex(v)))
        sum(x -> x[i], ∂fᵢ)
    end
end
loss(s, xs :: AbstractArray{T, 3}) where T =  (sum(x -> x^2, s(xs)) + T(2.0)*divergence(s, xs))/size(xs, 3)

score(ρ, x) = convert(eltype(x), sum(logpdf(ρ, x)))
propagate(x, t, Δt, b, D, s) = x + Δt * (b(x, t) + D(x, t)*s(x))

function plot_losses(losses)
    epochs = size(losses, 1)
    p = plot(vec(losses), title = "Score approximation", xaxis = "epochs", yaxis = "loss", label = "training loss")
    scatter!(p, 1:epochs:length(vec(losses)), vec(losses)[1:epochs:end], label = "discrete time propagation", marker = true)
end

function custom_train!(s, losses, xs; optimiser = Adam(5. * 10^-3), epochs = 25, record_losses = true, verbose = 1)
    θ = params(s)
    for epoch in 1:epochs
        loss_value, grads = withgradient(() -> loss(s, xs), θ)
        Flux.update!(optimiser, θ, grads)
        record_losses && (losses[epoch] = loss_value)
        verbose > 1 && println("Epoch $epoch, loss = $losses[epoch].")
    end
end

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ → Rᵈ
D   : Rᵈ → Rᵈˣᵈ
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm(xs :: Array{T, 3}, Δts, b, D, s; kwargs...) where T
    trajectories = zeros(T, size(xs)..., 1+length(Δts)) # trajectories[:, i, j, k] is particle i of sample j at time k
    trajectories[:, :, :, 1] = xs
    s_values = zeros(T, size(xs)..., 1+length(Δts))
    s_values[:, :, :, 1] = s(xs)
    losses = zeros(T, epochs, length(Δts))
    sbtm!(trajectories, losses, s_values, Δts, b, D, s; kwargs...)
    trajectories, losses, s_values
end

function sbtm!(trajectories :: Array{T, 4}, losses, s_values, Δts, b, D, s; verbose = 1, record_s_values = true, kwargs...) where T
    t = zero(eltype(Δts))
    for (k, Δt) in enumerate(Δts)
        verbose > 0 && println("Time $t")
        xs_k = @view trajectories[:, :, :, k]
        custom_train!(s, (@view losses[:, k]), xs_k; verbose = verbose, kwargs...)
        record_s_values && (s_values[:, :, :, k+1] = s(xs_k))
        trajectories[:, :, :, k+1] = propagate(xs_k, t, Δt, b, D, s)
        t += Δt
    end
end

function initialize_s(ρ₀, xs, size_hidden, num_hidden; activation = relu, verbose = 1, kwargs...)
    d = size(xs, 1)
    s = Chain(
        Dense(d => size_hidden, activation),
        repeat([Dense(size_hidden, size_hidden), activation], num_hidden)...,
        Dense(size_hidden => d))
    epochs = initialize_s!(s, ρ₀, xs; kwargs...)
    verbose > 0 && println("Took $epochs epochs to initialize. Initial loss: $(loss(s,xs))")
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

