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
function sbtm_landau(xs, ts; ρ₀ = nothing, s = nothing, size_hidden=100, num_hidden=1, kwargs...)
    isnothing(s) && isnothing(ρ₀) && error("Must provide either s or ρ₀.")
    isnothing(s) ? (s_new = initialize_s(ρ₀, xs, size_hidden, num_hidden; kwargs...)) : (s_new = deepcopy(s))
    solution = sbtm_landau_solve(Float32.(xs), Float32.(ts), s_new; kwargs...)
    solution
end

function sbtm_landau_solve(xs, ts :: AbstractVector{T}, s; epochs = 25, verbose = 0, optimiser = Adam(10^-4), kwargs...) where T
    tspan = (zero(T), ts[end])
    initial = xs
    # train s_ in a callback
    function affect!(integrator)
        xs = integrator.u
        state = Flux.setup(optimiser, s)
        @timeit "update NN" for epoch in 1:epochs
            loss_value, grads = withgradient(s -> loss(s, xs), s)
            Flux.update!(state, s, grads[1])
            verbose > 2 && println("Epoch $epoch, loss = $loss_value.")
        end
        verbose > 1 && println("loss = $(loss(s, xs)).")
    end
    cb = PresetTimeCallback(ts, affect!, save_positions=(false,false))
    
    s_values = similar(s(xs))
    z = similar(xs[:,1])
    v = similar(xs[:,1])
    p = (s, s_values, z, v)
    ode_problem = ODEProblem(landau_f!, initial, tspan, p)
    solution = solve(ode_problem, alg = Euler(), saveat=ts, tstops = ts, callback = cb)
    solution :: ODESolution
end

function landau_f!(dxs, xs, pars, t)
    s, s_values, z, v = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    s_values .= s(xs)
    @timeit "propagate particles" @views for p in 1:n, q in 1:n
        z .= xs[:,p] .- xs[:,q]
        v .= s_values[:,q] .- s_values[:,p]
        dotzv = fastdot(z,v)
        normsqz = normsq(z)
        v .*= normsqz
        v .-= dotzv .* z
        dxs[:,p] .+= v
    end
    dxs ./= 24*n
    nothing
end