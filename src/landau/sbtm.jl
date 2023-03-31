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
function sbtm(xs, ts, A; ρ₀ = nothing, s = nothing, size_hidden=100, num_hidden=1, kwargs...)
    isnothing(s) && isnothing(ρ₀) && error("Must provide either s or ρ₀.")
    isnothing(s) ? (s_new = initialize_s(ρ₀, xs, size_hidden, num_hidden; kwargs...)) : (s_new = deepcopy(s))
    solution = sbtm_solve(Float32.(xs), Float32.(ts), A, s_new; kwargs...)
    solution
end

function sbtm_solve(xs, ts :: AbstractVector{T}, A!, s; epochs = 25, verbose = 0, optimiser = Adam(10^-4), kwargs...) where T
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
    
    A_mat = zeros(eltype(xs), size(xs,1), size(xs,1))
    s_values = similar(s(xs))
    temp1 = similar(xs[:,1])
    temp2 = similar(xs[:,1])
    p = (A!, A_mat, s, s_values, temp1, temp2)
    ode_problem = ODEProblem(landau_f!, initial, tspan, p)
    solution = solve(ode_problem, alg = Euler(), saveat=ts, tstops = ts, callback = cb)
    solution :: ODESolution
end

function landau_f!(dxs, xs, pars, t)
    A!, A_mat, s, s_values, temp1, temp2 = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    s_values .= s(xs)
    @timeit "propagate particles" @views for p in 1:n, q in 1:n
        temp1 .= xs[:,p] .- xs[:,q]
        A!(A_mat, temp1)
        temp1 .= s_values[:,q] .- s_values[:,p]
        mul!(temp2, A_mat, temp1)
        dxs[:,p] .+= temp2
    end
    dxs ./= n
    nothing
end