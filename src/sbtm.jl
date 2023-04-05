# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, DifferentialEquations
using Distributions: MvNormal, logpdf
using Zygote: gradient, withgradient
using Flux.Optimise: Adam
using Flux: params


function initialize_s(ρ₀, xs, size_hidden, num_hidden; activation = relu, verbose = 0, kwargs...)
    d_bar = size(xs,1)
    s = Chain(
        Dense(d_bar => size_hidden, activation),
        repeat([Dense(size_hidden, size_hidden), activation], num_hidden-1)...,
        Dense(size_hidden => d_bar)
        )
    @timeit "NN init" epochs = initialize_s!(s, ρ₀, xs; verbose = verbose, kwargs...)
    verbose > 0 && println("Initialized the NN in $epochs epochs. Loss = $(loss(s, xs)).")
    return s
end

function initialize_s!(s, ρ₀, xs :: AbstractArray{T}; optimiser = Adam(10^-3), ε = T(10^-4), verbose = 0, record_s_values = false, record_losses = false) where T
    verbose > 1 && println("Initializing NN \n$s")
    ys = score(ρ₀, xs)
    ys_sum_squares = norm(ys)^2
    square_error(s) = norm(s(xs) - ys)^2 / ys_sum_squares
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

function loss(s, xs :: AbstractArray{T}, α = T(0.1)) where T
    ζ = randn(T, size(xs))
    denoise_val = ( s(xs .+ α .* ζ) ⋅ ζ - s(xs .- α .* ζ) ⋅ ζ ) / α
    (norm(s(xs))^2 + denoise_val)/num_particles(xs)
end

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ × R → Rᵈ
D   : Rᵈ × R → Rᵈˣᵈ or R
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm(xs, ts, b, D; ρ₀ = nothing, s = nothing, size_hidden=100, num_hidden=1, kwargs...)
    isnothing(s) && isnothing(ρ₀) && error("Must provide either s or ρ₀.")
    isnothing(s) ? (s_new = initialize_s(ρ₀, xs, size_hidden, num_hidden; kwargs...)) : (s_new = deepcopy(s))
    solution, s_values, losses = sbtm_solve(Float32.(xs), Float32.(ts), b, D, s_new; kwargs...)
    log = Dict("s_values" => s_values, "losses" => losses)
    solution
end

function sbtm_solve(xs, ts :: AbstractVector{T}, b, D, s; epochs = 25, record_s_values = false, record_losses = false, verbose = 0, optimiser = Adam(10^-4), kwargs...) where T
    tspan = (zero(T), ts[end])
    initial = xs
    s_values = zeros(T, size(xs)..., length(ts))
    record_s_values && (s_values[:, :, :, 1] = s(xs))
    losses = zeros(T, epochs, length(ts)-1)
    k = 1
    # train s_ in a callback
    function affect!(integrator)
        k += 1
        xs = integrator.u
        state = Flux.setup(optimiser, s)
        @timeit "NN train" for epoch in 1:epochs
            loss_value, grads = withgradient(s -> loss(s, xs), s)
            Flux.update!(state, s, grads[1])
            record_losses && (losses[epoch, k] = loss_value)
            verbose > 2 && println("Epoch $epoch, loss = $loss_value.")
        end
        record_s_values && (s_values[:, :, :, k] = s(xs))
    end
    cb = PresetTimeCallback(ts, affect!, save_positions=(false,false))
    
    p = (b, D, s)
    ode_problem = ODEProblem(fpe_f!, initial, tspan, p)
    solution = solve(ode_problem, alg = Euler(), saveat=ts, tstops = ts, callback = cb)
    solution, s_values, losses
end

function fpe_f!(dxs, xs, p, t) 
    b, D, s = p
    @timeit "propagate" dxs .= b(xs, t) .- D(xs, t) .* s(xs)
end