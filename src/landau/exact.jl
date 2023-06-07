using DifferentialEquations

include("../utils.jl")

"exact_score(t, xs) = ∇log ρ(t, xs)"
function exact_landau(xs, ts; exact_score, saveat, verbose = 0, kwargs...)
    verbose > 0 && println("Exact landau. n = $(num_particles(xs)).")
    tspan = (ts[1], ts[end])
    dt = ts[2] - ts[1]
    pars = exact_score
    
    ode_problem = ODEProblem(landau_f_exact!, xs, tspan, pars)
    solution = solve(ode_problem, saveat = saveat, alg = Euler(), dt = dt)
end

function landau_f_exact!(dxs, xs :: AbstractArray{T, 2}, pars, t) where T
    exact_score = pars
    score_values = exact_score(t, xs)
    @timeit "propagate particles" landau_3d_f!(dxs, xs, score_values)
end
