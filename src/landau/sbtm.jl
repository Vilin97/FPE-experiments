# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, DifferentialEquations
using Zygote: withgradient
using Flux.Optimise: Adam
using Flux: params

include("../sbtm.jl")

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ × R → Rᵈ
D   : Rᵈ × R → Rᵈˣᵈ or R
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm(xs, ts, A; ρ₀ = nothing, s = nothing, kwargs...)
    isnothing(s) && isnothing(ρ₀) && error("Must provide either s or ρ₀.")
    isnothing(s) ? (s_new = initialize_s(ρ₀, xs, 100, 1; kwargs...)) : (s_new = deepcopy(s))
    solution, s_values, losses = sbtm_solve(Float32.(xs), Float32.(ts), A, s_new; kwargs...)
    log = Dict("s_values" => s_values, "losses" => losses)
    solution
end

function sbtm_solve(xs, ts :: AbstractVector{T}, A, s; epochs = 25, record_s_values = false, record_losses = false, verbose = 0, optimiser = Adam(10^-4), kwargs...) where T
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
    
    p = (A, s)
    ode_problem = ODEProblem(landau_f!, initial, tspan, p)
    solution = solve(ode_problem, alg = Euler(), saveat=ts, tstops = ts, callback = cb)
    solution, s_values, losses
end

function landau_f!(dxs, xs, pars, t)
    A, s = pars
    n = num_particles(xs)
    @views for p in 1:n, q in 1:n
        xp = xs[:,p]
        xq = xs[:,q]
        dxs[:,p] .+= A(xp - xq)*(s(xq) - s(xp))
    end
end