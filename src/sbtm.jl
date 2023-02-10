# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra
using Distributions: MvNormal, logpdf
using Zygote: gradient, withgradient
using Flux.Optimise: Adam
using Flux: params


function initialize_s(ρ₀, xs, size_hidden, num_hidden; activation = relu, verbose = 0, kwargs...)
    d_bar, N, n = size(xs)
    d = d_bar*N
    s = Chain(
        # xs -> reshape(xs, d, n),
        Dense(d_bar => size_hidden, activation),
        repeat([Dense(size_hidden, size_hidden), activation], num_hidden-1)...,
        Dense(size_hidden => d_bar)
        # xs -> reshape(xs, d_bar, N, n)
        )
    @timeit "NN init" epochs = initialize_s!(s, ρ₀, xs; verbose = verbose, kwargs...)
    verbose > 0 && println("Done with NN initialization. Took $epochs epochs.")
    return s
end

function initialize_s!(s, ρ₀, xs :: AbstractArray{T, 3}; optimiser = Adam(10^-3), ε = T(10^-4), verbose = 1) where T
    verbose > 1 && println("Initializing NN \n$s")
    ys = gradient(x -> score(ρ₀, x), xs)[1]
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

function loss(s, xs :: AbstractArray{T, 3}, α = T(0.1)) where T
    ζ = randn(size(xs))
    denoise_val = ( dot(s(xs .+ α .* ζ), ζ) - dot(s(xs .- α .* ζ), ζ) ) / α
    (norm(s(xs))^2 + denoise_val)/size(xs, 3)
end
score(ρ, x) = convert(eltype(x), sum(logpdf(ρ, x)))
propagate(x, t, Δt, b, D, s) = x + Δt * (b(x, t) - D(x, t)*s(x))

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ × R → Rᵈ
D   : Rᵈ × R → Rᵈˣᵈ or R
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm(xs, ts, b, D; ρ₀ = nothing, s = nothing, kwargs...)
    isnothing(s) ? (s_new = initialize_s(ρ₀, xs, 100, 1; kwargs...)) : (s_new = deepcopy(s))
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
        for epoch in 1:epochs
            loss_value, grads = withgradient(s -> loss(s, xs), s)
            Flux.update!(state, s, grads[1])
            record_losses && (losses[epoch, k] = loss_value)
            verbose > 1 && println("Epoch $epoch, loss = $loss_value.")
        end
        record_s_values && (s_values[:, :, :, k] = s(xs))
    end
    cb = PresetTimeCallback(ts, affect!, save_positions=(false,false))
    function f!(dxs, xs, p, t) 
        dxs .= b(xs, t) .- D(xs, t) .* s(xs)
    end
    ode_problem = ODEProblem(f!, initial, tspan)
    solution = solve(ode_problem, alg = Euler(), saveat=ts, tstops = ts, callback = cb)
    solution, s_values, losses
end