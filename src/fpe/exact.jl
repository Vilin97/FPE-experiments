using DifferentialEquations

include("../utils.jl")

"exact_score(t, xs) = ∇log ρ(t, xs)"
function exact_fpe(xs, ts, b, D; exact_score, saveat, verbose = 0, kwargs...)
    verbose > 0 && println("Exact fpe. n = $(num_particles(xs)).")
    tspan = (ts[1], ts[end])
    dt = ts[2] - ts[1]
    pars = (exact_score, b, D)
    
    ode_problem = ODEProblem(exact_fpe_f!, xs, tspan, pars)
    solution = solve(ode_problem, saveat = saveat, alg = Euler(), dt = dt)
end

function exact_fpe_f!(dxs, xs :: AbstractArray{T, 2}, pars, t) where T
    exact_score, b, D = pars
    score_values = exact_score(t, xs)
    dxs .= -D(xs,t) .* score_values .+ b(xs, t)
end
