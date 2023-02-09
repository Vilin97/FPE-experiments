include("utils.jl")

using DifferentialEquations, LoopVectorization

function blob(xs, ts, b, D; ε = 1/π, kwargs...)
    T = typeof(ε)
    solution = blob_solve(T.(xs), T.(ts), b, D, ε)
    solution
end

function f!(dxs, xs_, pars, t)
    (ε, n, d, diff_norm2s, mol_sum, term1, term2, mols, b, D, d_bar) = pars
    xs = reshape(xs_, d, n)
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
    dxs .= reshape(-D(xs,t) .* term1 .+ term2, d_bar, :, n) .+ b(xs_, t)
end

function blob_solve(xs, ts :: AbstractVector{T}, b, D, ε :: T) where T
    tspan = (ts[1], ts[end])
    d_bar, N, n = size(xs)
    d = d_bar * N
    initial = xs
    diff_norm2s = zeros(T, n, n)
    mol_sum = zeros(T, n)
    term1 = zeros(T, d, n)
    term2 = zeros(T, d, n)
    mols = zeros(T, n, n)
    pars = (ε, n, d, diff_norm2s, mol_sum, term1, term2, mols, b, D, d_bar)
    
    ode_problem = ODEProblem(f!, initial, tspan, pars)
    solution = solve(ode_problem, saveat = ts, alg = Euler(), tstops = ts)
end