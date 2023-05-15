# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, DifferentialEquations
using Zygote: withgradient
using Flux.Optimise: Adam
using Flux: params
using TimerOutputs

include("../sbtm.jl")

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ × R → Rᵈ
D   : Rᵈ × R → Rᵈˣᵈ or R
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm_landau(xs, ts; ρ₀ = nothing, s = nothing, size_hidden=100, num_hidden=2, error_tolerance = 1e-3, kwargs...)
    isnothing(s) && isnothing(ρ₀) && error("Must provide either s or ρ₀.")
    isnothing(s) ? (s_new = initialize_s(ρ₀, Float32.(xs), size_hidden, num_hidden; error_tolerance = error_tolerance, kwargs...)) : (s_new = deepcopy(s))
    solution, models, losses = sbtm_landau_solve(Float32.(xs), Float32.(ts), s_new; kwargs...)
end

function sbtm_landau_solve(xs, ts :: AbstractVector{T}, s; saveat, epochs = 25, verbose = 0, optimiser = Adam(1e-4), denoising_alpha = T(0.4), record_models = false, record_losses = false, kwargs...) where T
    verbose > 0 && println("SBTM method. n = $(num_particles(xs)).")
    tspan = (zero(T), ts[end])
    dt = ts[2] - ts[1]
    initial = xs
    s_models = Vector{Chain}(undef, length(saveat))
    losses = zeros(T, epochs, length(ts))
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
        @timeit "update NN" for epoch in 1:epochs
            loss_value, grads = withgradient(s -> loss(s, xs, denoising_alpha), s)
            Flux.update!(state, s, grads[1])
            record_losses && (losses[epoch, k] = loss_value)
            verbose > 2 && println("Epoch $epoch, loss = $loss_value.")
        end
        verbose > 1 && println("Time $(integrator.t), loss = $(loss(s, xs, denoising_alpha)).")
    end
    cb = PresetTimeCallback(ts, affect!, save_positions=(false,false))
    
    s_values_temp = similar(s(xs))
    p = (s, s_values_temp)
    ode_problem = ODEProblem(landau_f_sbtm!, initial, tspan, p)
    solution = solve(ode_problem, alg = Euler(), saveat=saveat, dt = dt, callback = cb)
    solution :: ODESolution, s_models, losses
end

function landau_f_sbtm!(dxs, xs, pars, t)
    s, s_values = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    s_values .= s(xs)
    @timeit "propagate particles" @turbo for p = 1:n
        Base.Cartesian.@nexprs 3 i -> dx_i = zero(eltype(dxs))
        for q = 1:n
            dotzv = zero(eltype(dxs))
            normsqz = zero(eltype(dxs))
            Base.Cartesian.@nexprs 3 i -> begin
                z_i = xs[i, p] - xs[i, q]
                v_i = s_values[i, q] - s_values[i, p]
                dotzv += z_i * v_i
                normsqz += z_i * z_i
            end
            Base.Cartesian.@nexprs 3 i -> begin
                dx_i += v_i * normsqz - dotzv * z_i
            end
        end
        Base.Cartesian.@nexprs 3 i -> begin
            dxs[i, p] += dx_i
        end
    end
    dxs ./= 24*n
    nothing
end

landau_f! = landau_f_sbtm! # avoid JLD2 warnings