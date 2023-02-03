include("utils.jl")

using DifferentialEquations, LoopVectorization

function jhu(xs, Δts, b, D; ε = 1/π, kwargs...)
    T = typeof(ε)
    trajectories, solution = jhu_solve(T.(xs), T.(Δts), b, D, ε)
    trajectories, solution
end

function f!(dxs, xs, pars, t)
    (ε, n, d, diff_norm2s, mol_sum, term1, term2, mols, b, D, d_bar) = pars
    mol_sum .= zero(ε)
    diff_norm2s .= zero(ε)
    term1 .= zero(ε)
    term2 .= zero(ε)
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        diff_norm2s[p, q] += (xs[k, p] - xs[k, q])^2
    end
    @tturbo for p in 1:n, q in 1:n
        mols[p, q] = exp(-diff_norm2s[p, q]/ε)/sqrt((π*ε)^d)
        mol_sum[p] += mols[p, q]
    end
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        fac = -2. / ε * mols[p, q]
        diff_k = xs[k, p] - xs[k, q]
        term1[k, p] += fac * diff_k / mol_sum[p]
        term2[k, p] += fac * diff_k / mol_sum[q]
    end
    dxs .= -D(xs,t) .* term1 .+ term2 .+ reshape(b(reshape(xs, d_bar, :, n), t), d, n)
end

function jhu_solve(xs, Δts :: AbstractVector{T}, b, D, ε :: T) where T
    tspan = (zero(T), sum(Δts))
    ts = vcat(zero(T), cumsum(Δts))
    d_bar, N, n = size(xs)
    d = d_bar * N
    initial = reshape(xs, d, n)
    diff_norm2s = zeros(T, n, n)
    AA = zeros(T, n)
    term1 = zeros(T, d, n)
    term2 = zeros(T, d, n)
    mols = zeros(T, n, n)
    pars = (ε, n, d, diff_norm2s, AA, term1, term2, mols, b, D, d_bar)
    
    ode_problem = ODEProblem(f!, initial, tspan, pars)
    solution = solve(ode_problem, saveat = ts, alg = Euler(), tstops = ts)
    cat(solution.u..., dims=4), solution
end