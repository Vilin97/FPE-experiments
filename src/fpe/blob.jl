using DifferentialEquations, LoopVectorization, TimerOutputs, CUDA

include("../utils.jl")
include("../blob.jl")

function blob_fpe(xs, ts, b, D; ε :: T, kwargs...) where T
    solution = blob_fpe_solve(T.(xs), T.(ts), b, D, ε; kwargs...)
end

function blob_fpe_solve(xs, ts :: AbstractVector{T}, b, D, ε :: T; saveat, verbose = 0, kwargs...) where T
    verbose > 0 && println("Blob method. n = $(num_particles(xs)), ε = $ε.")
    tspan = (ts[1], ts[end])
    score_params_ = score_params(xs, ε)
    score_values_temp = zero(xs)
    pars = (score_values_temp, b, D, score_params_)
    
    ode_problem = ODEProblem(blob_fpe_f!, xs, tspan, pars)
    solution = solve(ode_problem, saveat = saveat, alg = Euler(), tstops = ts)
end

function blob_fpe_f!(dxs, xs_, pars, t)
    score_values, b, D, score_params = pars
    n = num_particles(xs_)
    d = get_d(xs_)
    xs = reshape(xs_, d, n)
    @timeit "compute score" blob_score!(score_values, xs, score_params)
    dxs .= reshape(-D(xs_,t) .* score_values, size(xs_)) .+ b(xs_, t)
end

