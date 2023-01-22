# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra
using Distributions: MvNormal, logpdf
using Zygote: gradient, withgradient
using Flux.Optimise: Adam
using Flux: params


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
    println("Done with NN initialization. Took $epochs epochs and $t seconds.")
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

# approximate divergence of f at v
function denoise(s, xs :: AbstractArray{T, 3}, ζ, α = T(0.1)) where T
    ζ_ = reshape(rand(ζ), size(xs))
    return ( sum(s(xs .+ α .* ζ_) .* ζ_) - sum(s(xs .- α .* ζ_) .* ζ_) ) / (T(2.)*α)
end

loss(s, xs :: AbstractArray{T, 3}, ζ) where T =  (norm(s(xs))^2 + T(2.0)*denoise(s, xs, ζ))/size(xs, 3)
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
function sbtm(xs, Δts, b, D; ρ₀, s = nothing, kwargs...)
    isnothing(s) && (s = initialize_s(ρ₀, xs, 100, 1))
    ts = vcat(zero(T), cumsum(Δts))
    trajectories, losses, s_values, solution = sbtm_solve(Float32.(xs), Float32.(ts), b, D, deepcopy(s); kwargs...)
    extras = Dict(["losses" => losses, "s_values" => s_values, "solution" => solution])
    trajectories, extras
end

# Need to use an explicit method, because the NN will be retrained every time f is called
function sbtm_solve(xs, ts :: AbstractVector{T}, b, D, s; epochs = 25, record_s_values = false, record_losses = false, verbose = 0, optimiser = Adam(10^-4)) where T
    tspan = (zero(T), ts[end])
    initial = xs
    s_values = zeros(T, size(xs)..., length(ts))
    record_s_values && (s_values[:, :, :, 1] = s(xs))
    losses = zeros(T, epochs, length(ts)-1)
    θ = params(s)
    k = 1
    ζ = MvNormal(zeros(T, length(xs)), I(length(xs)))
    function f(xs, p, t)
        k += 1
        for epoch in 1:epochs
            loss_value, grads = withgradient(() -> loss(s, xs, ζ), θ)
            Flux.update!(optimiser, θ, grads)
            record_losses && (losses[epoch, k] = loss_value)
            verbose > 1 && println("Epoch $epoch, loss = $loss_value.")
        end
        record_s_values && (s_values[:, :, :, k] = s(xs))
        b(xs, t) - D(xs, t)*s(xs)
    end
    ode_problem = ODEProblem(f, initial, tspan)
    solution = solve(ode_problem, Euler(), saveat=ts, tstops = ts)
    trajectories = cat(solution.u..., dims=4)
    trajectories, losses, s_values, solution
end

# using a callback to train the NN. Does not work...
function sbtm_solve1(xs, Δts :: AbstractVector{T}, b, D, s; epochs = 25, record_s_values = false, record_losses = false, verbose = 0, optimiser = Adam(10^-4)) where T
    s_ = deepcopy(s)
    tspan = (zero(T), sum(Δts))
    ts = vcat(zero(T), cumsum(Δts))
    initial = xs
    s_values = zeros(T, size(xs)..., 1+length(Δts))
    record_s_values && (s_values[:, :, :, 1] = s_(xs))
    losses = zeros(T, epochs, length(Δts))
    θ = params(s_)
    k = 1
    ζ = MvNormal(zeros(T, d), I(d))
    p = (k, s_, θ, record_losses, record_s_values, s_values, losses, epochs, optimiser, verbose, ζ)
    # train s_ in a callback
    function affect!(integrator)
        k, s_, θ, record_losses, record_s_values, s_values, losses, epochs, optimiser, verbose, ζ = integrator.p
        k += 1
        for epoch in 1:epochs
            loss_value, grads = withgradient(() -> loss(s_, xs, ζ), θ)
            Flux.update!(optimiser, θ, grads)
            record_losses && (losses[epoch, k] = loss_value)
            verbose > 1 && println("Epoch $epoch, loss = $loss_value.")
        end
        record_s_values && (s_values[:, :, :, k] = s_(xs))
        integrator.p = (k, s_, θ, record_losses, record_s_values, s_values, losses, epochs, optimiser, verbose, ζ)
    end
    cb = PresetTimeCallback(ts, affect!, save_positions=(false,false))
    f(xs, p, t) = b(xs, t) - D(xs, t)*p[2](xs)
    ode_problem = ODEProblem(f, initial, tspan, p)
    solution = solve(ode_problem, Euler(), saveat=ts, tstops = ts, callback = cb)
    trajectories = cat(solution.u..., dims=4)
    trajectories, losses, s_values, solution
end