using DifferentialEquations, LoopVectorization, TimerOutputs
include("../utils.jl")
include("../blob.jl")

function blob_landau(xs, ts; ε=0.025, kwargs...)
    T = typeof(ε)
    solution = blob_landau_solve(T.(xs), T.(ts), ε; kwargs...)
end

function blob_landau_solve(xs, ts :: AbstractVector{T}, ε :: T; saveat, verbose = 0, kwargs...) where T
    verbose > 0 && println("Blob method landau. n = $(num_particles(xs)), ε = $ε.")
    tspan = (ts[1], ts[end])
    score_params_ = score_params(xs, ε)
    score_values_temp = zero(xs)
    pars = (score_values_temp, score_params_)
    
    ode_problem = ODEProblem(landau_f_blob!, xs, tspan, pars)
    solution = solve(ode_problem, saveat = saveat, alg = Euler(), tstops = ts)
end

function landau_f_blob!(dxs, xs, pars, t)
    score_values, score_params = pars
    @timeit "compute score" blob_score!(score_values, xs, score_params)
    @timeit "propagate particles" landau_3d_f!(dxs, xs, score_values)
end