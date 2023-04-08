using DifferentialEquations, LoopVectorization

function blob_score!(score_array, xs, pars)
    (ε, diff_norm2s, mol_sum, term1, term2, mols) = pars
    d, n = size(xs)
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
    score_array .= term1 .+ term2
end

function landau_f_blob!(dxs, xs, pars, t)
    score_values, z, v, score_params = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    @timeit "compute score" blob_score!(score_values, xs, score_params)
    @timeit "propagate particles" @views for p in 1:n, q in 1:n
        z .= xs[:,p] .- xs[:,q]
        v .= score_values[:,q] .- score_values[:,p]
        dxs[:,p] .+=  v .* normsq(z) .- fastdot(z,v) .* z
    end
    dxs ./= 24*n
    nothing
end

function blob_landau(xs, ts; ε = 1/π, kwargs...)
    T = typeof(ε)
    solution = blob_landau_solve(T.(xs), T.(ts), ε)
end

function blob_landau_solve(xs, ts :: AbstractVector{T}, ε :: T) where T
    tspan = (ts[1], ts[end])
    d, n = size(xs)
    initial = xs
    diff_norm2s = zeros(T, n, n)
    mol_sum = zeros(T, n)
    term1 = zeros(T, d, n)
    term2 = zeros(T, d, n)
    mols = zeros(T, n, n)
    score_params = (ε, diff_norm2s, mol_sum, term1, term2, mols)

    score_values_temp = similar(s(xs))
    z = similar(xs[:,1])
    v = similar(xs[:,1])
    pars = (score_values_temp, z, v, score_params)
    
    ode_problem = ODEProblem(landau_f_blob!, initial, tspan, pars)
    solution = solve(ode_problem, saveat = ts, alg = Euler(), tstops = ts)
end