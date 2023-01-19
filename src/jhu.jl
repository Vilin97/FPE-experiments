using Zygote

include("utils.jl")

# solve using DifferentialEquations
using DifferentialEquations

function jhu(xs, Δts, b, D, ε :: T) where T
    jhu_solve(T.(xs), T.(Δts), b, D, ε)
end

function jhu_solve(xs, Δts :: AbstractVector{T}, b, D, ε) where T
    tspan = (zero(T), sum(Δts))
    ts = vcat(zero(T), cumsum(Δts))
    initial = xs
    function f(xs, p, t)
        d_bar, N, n = size(xs)
        flat_xs = reshape(xs, d_bar*N, n)
        xps = eachslice(flat_xs, dims=2)
        m = [mol(ε, x_p - x_q) for x_q in xps, x_p in xps]
        g = [gradient(x -> mol(ε, x - x_q), x_p)[1] for x_q in xps, x_p in xps]
        M = reshape(sum(m, dims=1), :)
        G = reshape(sum(g, dims=1), :)
        d1 = reduce(hcat, G ./ M)
        d2 = reduce(hcat, g * (one(eltype(M)) ./ M))
        b(xs, t) - D(xs, t) * reshape(d1 + d2, d_bar, N, n)
    end
    solution = solve(ODEProblem(f, initial, tspan), saveat = ts)
    solution, cat(solution.u..., dims=4)
end