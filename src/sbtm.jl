# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, Plots
using Distributions: MvNormal, logpdf
using Zygote: gradient, withgradient
using Flux.Optimise: Adam
using Flux: params

# approximate divergence of f at v
function denoise(s, xs :: AbstractArray{T, 3}, α = T(0.1)) where T
    d = length(xs)
    ζ = reshape(rand(MvNormal(zeros(T, d), I(d))), size(xs))
    return ( sum(s(xs .+ α .* ζ) .* ζ) - sum(s(xs .- α .* ζ) .* ζ) ) / (T(2.)*α)
end

loss(s, xs :: AbstractArray{T, 3}) where T =  (sum(x -> x^2, s(xs)) + T(2.0)*denoise(s, xs))/size(xs, 3)

score(ρ, x) = convert(eltype(x), sum(logpdf(ρ, x)))
propagate(x, t, Δt, b, D, s) = x + Δt * (b(x, t) - D(x, t)*s(x))

function plot_losses(losses)
    epochs = size(losses, 1)
    p = plot(vec(losses), title = "Score approximation", xaxis = "epochs", yaxis = "loss", label = "training loss")
    scatter!(p, 1:epochs:length(vec(losses)), vec(losses)[1:epochs:end], label = "discrete time propagation", marker = true)
end

function custom_train!(s, losses, xs; optimiser = Adam(10^-4), epochs, record_losses = true, verbose = 1)
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
function sbtm(xs :: Array{T, 3}, Δts, b, D, s; epochs = 25, kwargs...) where T
    trajectories = zeros(T, size(xs)..., 1+length(Δts)) # trajectories[:, i, j, k] is particle i of sample j at time k
    trajectories[:, :, :, 1] = xs
    s_values = zeros(T, size(xs)..., 1+length(Δts))
    s_values[:, :, :, 1] = s(xs)
    losses = zeros(T, epochs, length(Δts))
    sbtm!(trajectories, losses, s_values, Δts, b, D, s; epochs = epochs, kwargs...)
    trajectories, losses, s_values
end

function sbtm!(trajectories :: Array{T, 4}, losses, s_values, Δts, b, D, s; epochs, verbose = 1, record_s_values = true, kwargs...) where T
    t = zero(eltype(Δts))
    for (k, Δt) in enumerate(Δts)
        verbose > 0 && println("Time $t")
        xs_k = @view trajectories[:, :, :, k]
        custom_train!(s, (@view losses[:, k]), xs_k; epochs = epochs, verbose = verbose, kwargs...)
        record_s_values && (s_values[:, :, :, k+1] = s(xs_k))
        trajectories[:, :, :, k+1] = propagate(xs_k, t, Δt, b, D, s)
        t += Δt
    end
end

function initialize_s!(s, ρ₀, xs :: AbstractArray{T, 3}; optimiser = Adam(10^-4), ε = T(10^-4), verbose = 1) where T
    verbose > 1 && println("Initializing NN \n$s")
    ys = gradient(x -> score(ρ₀, x), xs)[1]
    ys_sum_squares = sum(ys.^2)
    square_error(s) = sum( (s(xs) - ys).^2 / ys_sum_squares )
    epoch = 0
    θ = params(s)
    loss_value = ε + one(T)
    while loss_value > ε
        loss_value, grads = withgradient(() -> square_error(s), θ)
        Flux.update!(optimiser, θ, grads)
        epoch += 1
        verbose > 1 && epoch % 1000 == 0 && println("Epoch $epoch, loss $loss_value")
    end
    epoch
end

