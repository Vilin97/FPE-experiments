include("utils.jl")

using DifferentialEquations

function jhu(xs, Δts, b, D; ε = 1/π, ρ₀ = nothing)
    T = typeof(ε)
    trajectories, solution = jhu_solve(T.(xs), T.(Δts), b, D, ε)
    trajectories, solution
end

function jhu_solve(xs, Δts :: AbstractVector{T}, b, D, ε) where T
    tspan = (zero(T), sum(Δts))
    ts = vcat(zero(T), cumsum(Δts))
    initial = xs
    d_bar, N, n = size(xs)
    dv_dt_norms = zeros(T, size(ts))
    # k = 0
    function f(xs, dv_dt_norms, t)
        # k += 1
        flat_xs = reshape(xs, d_bar*N, n)
        xps = eachslice(flat_xs, dims=2)
        m = [mol(ε, x_p - x_q) for x_q in xps, x_p in xps]
        g = [grad_mol(ε, x_p-x_q) for x_q in xps, x_p in xps]
        M = reshape(sum(m, dims=1), :)
        G = reshape(sum(g, dims=1), :)
        d1 = reduce(hcat, G ./ M)
        d2 = reduce(hcat, g * (one(eltype(M)) ./ M))
        dv_dt = b(xs, t) - D(xs, t) * reshape(d1 + d2, d_bar, N, n)
        # dv_dt_norms[k] = norm(dv_dt)
        dv_dt
    end
    ode_problem = ODEProblem(f, initial, tspan, dv_dt_norms)
    solution = solve(ode_problem, saveat = ts)
    cat(solution.u..., dims=4), solution
end