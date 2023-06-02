using Flux, DifferentialEquations, TimerOutputs
using Zygote: gradient, withgradient
using Flux.Optimise: Adam
using Random: randn!

include("../utils.jl")
include("../sbtm.jl")

function sbtm_fpe(xs, ts, b, D; ρ₀ = nothing, s = nothing, size_hidden=100, num_hidden=1, error_tolerance = 1e-3, usegpu = false, kwargs...)
    if usegpu
        xs = gpu(xs)
        s = gpu(s)
    else
        xs = Float32.(xs)
    end
    isnothing(s) && isnothing(ρ₀) && error("Must provide either s or ρ₀.")
    isnothing(s) ? (s_new = initialize_s(ρ₀, xs, size_hidden, num_hidden; error_tolerance = error_tolerance, kwargs...)) : (s_new = deepcopy(s))
    solution, models, losses = sbtm_fpe_solve(xs, ts, b, D, s_new; kwargs...)
end

function sbtm_fpe_solve(xs :: AbstractArray{T}, ts, b, D, s; saveat, epochs = 25, verbose = 0, optimiser = Adam(1e-4), denoising_alpha = T(0.4), record_models = false, record_losses = false, kwargs...) where T
    verbose > 0 && println("SBTM method fpe. n = $(num_particles(xs)).")
    tspan = (zero(T), ts[end])
    dt = ts[2] - ts[1]
    initial = xs
    s_models = Vector{Chain}(undef, length(saveat))
    losses = zeros(T, epochs, length(ts))
    ζ = similar(xs)
    # train s_ in a callback
    function affect!(integrator)
        DifferentialEquations.u_modified!(integrator, false)
        k = integrator.iter + 1
        xs = integrator.u
        if record_models
            idx = findfirst(x -> x ≈ integrator.t, saveat)
            if idx !== nothing 
                s_models[idx] = deepcopy(s)
            end
        end
        state = Flux.setup(optimiser, s)
        for epoch in 1:epochs
            randn!(ζ)
            loss_value, grads = withgradient(s -> loss(s, xs, ζ, denoising_alpha), s)
            Flux.update!(state, s, grads[1])
            record_losses && (losses[epoch, k] = loss_value)
            verbose > 2 && println("Epoch $epoch, loss = $loss_value.")
        end
        verbose > 1 && println("Time $(integrator.t), loss = $(loss(s, xs, denoising_alpha)).")
    end
    cb = PresetTimeCallback(ts, affect!, save_positions=(false,false))
    
    p = (b, D, s)
    ode_problem = ODEProblem(sbtm_fpe_f!, initial, tspan, p)
    solution = solve(ode_problem, alg = Euler(), saveat=saveat, dt = dt, callback = cb)
    solution, s_models, losses
end

function sbtm_fpe_f!(dxs, xs, p, t) 
    b, D, s = p
    dxs .= b(xs, t) .- D(xs, t) .* s(xs)
end

