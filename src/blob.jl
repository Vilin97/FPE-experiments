include("utils.jl")

using DifferentialEquations, LoopVectorization, Zygote
using Random: seed!

function blob(xs, ts, b, D; ε = 1/π, kwargs...)
    T = typeof(ε)
    solution = blob_solve(T.(xs), T.(ts), b, D, ε)
    solution
end

function blob_f!(dxs, xs_, pars, t)
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
    
    ode_problem = ODEProblem(blob_f!, initial, tspan, pars)
    solution = solve(ode_problem, saveat = ts, alg = Euler(), tstops = ts)
end

using Distributions, LinearAlgebra, Optimisers, TimerOutputs

function set_epsilon(ρ, xs; default_eps = 0.5, optimiser = Optimisers.Adam(10^-2), max_epochs = 10^3)
    ε = [default_eps]
    state = Optimisers.setup(optimiser, ε)
    ys = score(ρ, xs)
    ys_sum_squares = norm(ys)^2
    prev_loss = -Inf
    for epoch in 1:max_epochs
        @timeit "compute gradient" loss_value, grads = withgradient(ε -> square_error(ε, ys, xs, ys_sum_squares), ε)
        epoch % 10 == 0 && println("Epoch $epoch: loss = $(round(loss_value, digits=4)), ε = $(round(ε[1], digits=4)).")
        if abs(loss_value - prev_loss) < 1e-5
            println("Converged to $(round(ε[1], digits=4)) in $epoch epochs.")
            return ε[1]
        end
        Optimisers.update!(state, ε, grads[1])
        prev_loss = loss_value
    end
    println("Did not converge in $max_epochs. ε = $(round(ε[1], digits=3)).")
    return ε[1]
end

# function s(ε, x, xs)
#     sum(grad_mol(ε[1], x - xp) for xp in eachcol(xs))/sum(mol(ε[1], x - xp) for xp in eachcol(xs))
# end
function square_error(ε_, ys, xs, ys_sum_squares)
    ε = ε_[1]
    d,n = size(xs)
    @views diff_norm2s = [norm(xs[:, p] .- xs[:, q])^2 for p in 1:n, q in 1:n]
    @views mols = exp.(-diff_norm2s/ε)/sqrt((π*ε)^d)
    mol_grad_sum = hcat([sum([-2. / ε * mols[p, q] .* (xs[:, q] .- xs[:, p]) for p in 1:n]) for q in 1:n]...)
    mol_sum = reshape(sum(mols, dims=2), 1,n)
    norm(mol_grad_sum ./ mol_sum - ys)^2/ys_sum_squares
end

# TODO this is incorrect. But probably not worth spending more time on it.
# function d_square_error_d_eps(ε_, ys, xs, ys_sum_squares)
#     ε = ε_[1]
#     d,n = size(xs)
#     res = 0.
#     @views for (q, xq) in enumerate(eachcol(xs))
#         diff_norm2s = [norm(xs[:, p] .- xq)^2 for p in 1:n]
#         mols = exp.(-diff_norm2s/ε)/sqrt((π*ε)^d)
#         mols_sum = sum(mols)
#         grad_mols_sum = sum([-2. / ε * mols[p] .* (xq .- xs[:, p]) for p in 1:n])
#         dmols_sum = sum([mols[p] * (norm(xq - xs[:, p])^2 / (2ε^2) - d/(2ε)) for p in 1:n])
#         dgrad_mols_sum = sum([(xq - xs[:, p]) .* mols[p]/ε^2 * (d+1 - norm(xq - xs[:, p])^2/(2ε)) for p in 1:n])
#         dscore = (dgrad_mols_sum .* mols_sum - grad_mols_sum .* dmols_sum) ./ mols_sum^2
#         res += (grad_mols_sum ./ mols_sum - ys[:, q])' * dscore
#     end
#     res/ys_sum_squares
# end

d,n = 2,20
ρ = MvNormal(I(d))
seed!(1235)
xs = rand(ρ, n)
# reset_timer!()
# @timeit "learn epsilon" set_epsilon(ρ, xs)
# print_timer()


ys = score(ρ, xs)
ε = [0.5]
ys_sum_squares = norm(ys)^2
d_square_error_d_eps(ε, ys, xs, ys_sum_squares)
@btime d_square_error_d_eps(ε, ys, xs, ys_sum_squares)
@btime withgradient(ε -> square_error(ε, ys, xs, ys_sum_squares), ε)