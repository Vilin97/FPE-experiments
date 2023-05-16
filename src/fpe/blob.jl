using DifferentialEquations, LoopVectorization, TimerOutputs, CUDA

include("../utils.jl")
include("../blob.jl")

function blob_fpe(xs, ts, b, D; ε, usegpu = false, kwargs...)
    if usegpu
        xs = cu(xs)
        ε = Float32(ε)
    end
    solution = blob_fpe_solve(xs, ts, b, D, ε; kwargs...)
end

function blob_fpe_solve(xs :: AbstractArray{T}, ts, b, D, ε :: T; saveat, verbose = 0, kwargs...) where T
    verbose > 0 && println("Blob method fpe. n = $(num_particles(xs)), ε = $ε.")
    tspan = (ts[1], ts[end])
    dt = ts[2] - ts[1]
    score_params_ = score_params(xs, ε)
    score_values_temp = zero(xs)
    pars = (score_values_temp, b, D, score_params_)
    
    ode_problem = ODEProblem(blob_fpe_f!, xs, tspan, pars)
    solution = solve(ode_problem, saveat = saveat, alg = Euler(), dt = dt)
end

function blob_fpe_f!(dxs, xs_, pars, t)
    score_values, b, D, score_params = pars
    n = num_particles(xs_)
    d = get_d(xs_)
    xs = reshape(xs_, d, n)
    blob_score!(score_values, xs, score_params)
    dxs .= -D(xs_,t) .* reshape(score_values, size(xs_)) .+ b(xs_, t)
end

function blob_fpe_f!(dxs, xs :: AbstractArray{T, 2}, pars, t) where T
    score_values, b, D, score_params = pars
    blob_score!(score_values, xs, score_params)
    dxs .= -D(xs,t) .* score_values .+ b(xs, t)
end