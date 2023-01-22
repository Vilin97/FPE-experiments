# analyze the trajectories from the first example in the paper

using DifferentialEquations, JLD2, Distributions, ComponentArrays, LinearAlgebra, Plots
using Random: seed!
include("utils.jl")
plotly()

# reconstruct the initial parameters
function initial_params(N, num_samples, num_timestamps; seed = N*num_samples*num_timestamps)
    seed!(seed)
    d_bar = 2
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)
    ts = vcat(0f0, cumsum(Δts))
    xs, Δts, b, D, ρ₀, target, a, w, α, β, ts, d_bar
end

### ANALYTICAL ###
function solve_analytically(N, num_samples, num_timestamps)
    xs, Δts, b, D, ρ₀, target, a, w, α, β, ts, d_bar = initial_params(N, num_samples, num_timestamps)
    println("Solving analytically")
    tspan = (0f0, sum(Δts))
    p = (α, size(xs, 2), t -> D(xs, t), β) # assumes D is constant in space
    initial = ComponentVector(m = mean(ρ₀), Cd = cov(ρ₀), Co = zeros(eltype(xs), 2, 2))
    function f!(dcov, cov, p, t :: T) where T
        α, N, D_, β = p
        dcov.m = β(t) - cov.m
        dcov.Cd = T(2)*(α-one(T))*cov.Cd - T(2)*α/N*(cov.Cd + (N-1)*cov.Co) + T(2)*D_(t)*I
        dcov.Co = T(2)*(α-one(T))*cov.Co - T(2)*α/N*(cov.Cd + (N-1)*cov.Co)
    end
    analytic_solution = solve(ODEProblem(f!, initial, tspan, p), saveat = ts)
    ρ(t) = MvNormal(kron(ones(N), analytic_solution(t).m), kron((analytic_solution(t).Cd - analytic_solution(t).Co), I(N)) + kron(analytic_solution(t).Co, ones(Float32, N, N)))
    return analytic_solution, ρ
end

function analytic_entropies(ρ, ts)
    entropy.(ρ.(ts))
end

function analytic_moments(ρ, ts)
    mean.(ρ.(ts)), cov.(ρ.(ts))
end

"analytic marginal pdf at points xs"
function analytic_marginals(xs, time_index, analytic_solution)
    d_bar = 2
    sol = analytic_solution[time_index]
    marginal_dist = MvNormal(sol.m, I(d_bar)*(sol.Cd - sol.Co)[1] + ones(d_bar, d_bar)*sol.Co[1])
    pdf(marginal_dist, xs)
end

### EMPIRICAL ###
function empirical_entropies(ε, trajectories)
    d_bar, N, n, _ = size(trajectories)
    flat_trajectories = reshape(trajectories, d_bar*N, n, :)
    [-sum( log(Mol(ε, x, flat_xs)/n) for x in eachslice(flat_xs, dims=2) )/n for flat_xs in eachslice(flat_trajectories, dims=3)]
end

"empirical marginal pdf at point x ∈ R²"
function empirical_marginal(ε, x, xs :: AbstractArray{T, 3}) where T
    Mol(ε, x, (@view xs[:,1,:]))/size(xs, 3)
end

empirical_first_moment(flat_xs :: AbstractArray{T, 2}) where T = vec(mean(flat_xs, dims = 2))
empirical_second_moment(flat_xs :: AbstractArray{T, 2}) where T = flat_xs * flat_xs' / (size(flat_xs, 2) - 1)
empirical_covariance(flat_xs :: AbstractArray{T, 2}) where T = empirical_second_moment(flat_xs .- empirical_first_moment(flat_xs))

function empirical_moments(trajectories)
    d_bar, N, num_samples, _ = size(trajectories)
    flat_trajectories = reshape(trajectories, d_bar*N, num_samples, :)
    expectations = [empirical_first_moment(flat_xs) for flat_xs in eachslice(flat_trajectories, dims=3)]
    covariances = [empirical_covariance(flat_xs) for flat_xs in eachslice(flat_trajectories, dims=3)]
    expectations, covariances
end

function make_plots(trajectories, labels, analytic_solution, ρ; ε_entropy = 1.24, ε_marginal = 0.15)
    println("Making plots")
    d_bar, N, num_samples, _ = size(trajectories[1])
    _, Δts, b, D, ρ₀, target, a, w, α, β, ts, _ = initial_params(N, num_samples, size(trajectories[1], 4)-1)
    analytic_expectations, analytic_covariances = analytic_moments(ρ, ts)
    analytic_entropies_ = analytic_entropies(ρ, ts)

    # for heatmap plotting
    range = -3:0.1:3
    x_grid = Iterators.product(range, range)
    time_indices = 1:size(trajectories[1], 4)÷4:size(trajectories[1], 4)
    heatmaps = Matrix{Any}(undef, length(trajectories)+1, length(time_indices))
    analytic_marginals_ = [analytic_marginals(collect.(x_grid), time_index, analytic_solution) for time_index in time_indices]

    heatmaps[1, :] = [heatmap(range, range, analytic_marginals_[i]', title = "analytic marginal at t = $(round(t, digits = 2))") for (i, t) in enumerate(ts[time_indices])]

    p1 = plot(title = "mean position comparison, N = $(N), num_samples = $(num_samples)", ylabel = "mean position 2-norm")
    plot!(p1, ts, norm.(analytic_expectations), label = "analytic")

    p2 = plot(title = "covariance trace comparison", ylabel = "covariance trace")
    plot!(p2, ts, tr.(analytic_covariances), label = nothing)

    p3 = plot(title = "covariance norm comparison", ylabel = "covariance 1-norm")
    plot!(p3, ts, norm.(analytic_covariances, 1), label = nothing)

    p4 = plot(title = "entropy comparison", ylabel = "entropy")
    plot!(p4, ts, analytic_entropies_, label = nothing)

    for (j, trajectory) in enumerate(trajectories)
        expectations, covariances = empirical_moments(trajectory)
        entropies = empirical_entropies(ε_entropy, trajectory)

        plot!(p1, ts, norm.(expectations), label = "$(labels[j])")
        plot!(p2, ts, tr.(covariances), label = nothing)
        plot!(p3, ts, norm.(covariances, 1), label = nothing)
        plot!(p4, ts, entropies, label = nothing)
        
        empirical_marginals = [[empirical_marginal(ε_marginal, [x...], xs) for x in x_grid] for xs in collect(eachslice(trajectory, dims=4))[time_indices]]
        
        for (i, t) in enumerate(ts[time_indices])
            heatmaps[j+1, i] = heatmap(range, range, empirical_marginals[i]', title = "marginal at t = $(round(t, digits = 2)), $(labels[j])")
        end
    end
    heatmap_plot = plot(heatmaps..., layout = (size(heatmaps, 2), size(heatmaps, 1)), size = (2000, 1300))
    stats_plot = plot(p1, p2, p3, p4, layout = (4, 1), size = (1000, 1500))
    stats_plot, heatmap_plot
end

# epsilon experiment:
function jhu_epsilon_experiment()
    epsilons = 0.1:0.02:0.2
    rounded_εs = round.(epsilons, digits = 2)
    trajectories = [JLD2.load("jhu_eps_experiment//jhu_epsilon_experiment_eps_$(rounded_ε).jld2")["jhu_trajectories"] for rounded_ε in rounded_εs]
    labels = ["ε = $(rounded_ε)" for rounded_ε in rounded_εs]
    stats_plot, heatmap_plot = make_plots(trajectories, labels)
    stats_plot, heatmap_plot
end

# sbtm vs jhu
function sbtm_vs_jhu_experiment(N, num_samples, num_timestamps)
    println("Loading data for N = $N, num_samples = $num_samples, num_timestamps = $num_timestamps")
    seed = N*num_samples*num_timestamps
    seed!(seed)
    data_jhu = JLD2.load("data/moving_trap_jhu_$seed.jld2")
    data_sbtm = JLD2.load("data/moving_trap_sbtm_$seed.jld2")
    @assert data_jhu["N"] == data_sbtm["N"] && data_jhu["num_samples"] == data_sbtm["num_samples"] && data_jhu["num_timestamps"] == data_sbtm["num_timestamps"] && data_jhu["N"] == N && data_jhu["num_samples"] == num_samples && data_jhu["num_timestamps"] == num_timestamps
    jhu_trajectories = data_jhu["trajectories"]
    sbtm_trajectories = data_sbtm["trajectories"]
    trajectories = [sbtm_trajectories, jhu_trajectories]
    labels = ["sbtm", "jhu"]

    analytic_solution, ρ = solve_analytically(N, num_samples, num_timestamps)
    stats_plot, heatmap_plot = make_plots(trajectories, labels, analytic_solution, ρ)
end

function sbtm_new_old_experiment(N, num_samples, num_timestamps)
    println("Loading data for N = $N, num_samples = $num_samples, num_timestamps = $num_timestamps")
    seed = N*num_samples*num_timestamps
    data_sbtm_old = JLD2.load("old_new_sbtm/moving_trap_sbtm_$seed.jld2")
    data_sbtm_new = JLD2.load("old_new_sbtm/moving_trap_sbtm_new_$seed.jld2")
    sbtm_old_trajectories = data_sbtm_old["trajectories"]
    sbtm_new_trajectories = data_sbtm_new["trajectories"]
    trajectories = [sbtm_old_trajectories, sbtm_new_trajectories]
    labels = ["sbtm_old", "sbtm_new"]

    analytic_solution, ρ = solve_analytically(N, num_samples, num_timestamps)
    stats_plot, heatmap_plot = make_plots(trajectories, labels, analytic_solution, ρ)
end

# no_drift_experiment
function no_drift_experiment()
    data = JLD2.load("no_drift_experiment/moving_trap_data_$(1000000).jld2")
    sbtm_trajectories = data["sbtm_trajectories"]  
    jhu_trajectories = data["jhu_trajectories"] 
    trajectories = [sbtm_trajectories, jhu_trajectories]
    labels = ["sbtm", "jhu"]
    stats_plot, heatmap_plot = make_plots(trajectories, labels)
    stats_plot, heatmap_plot
end

function print_mol_stats(N, num_samples, num_timestamps, t = 0.)
    seed = N*num_samples*num_timestamps
    seed!(seed)
    xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps)

    εs = 1/(π) .* (0.6:0.2:1.4)

    drift = b(xs, t)
    d_bar, N, n = size(xs)
    flat_xs = Float64.(reshape(xs, d_bar*N, n))
    xps = eachslice(flat_xs, dims=2)
    pairwise_distances = [norm(x_p - x_q) for x_q in xps, x_p in xps]

    s_value = JLD2.load("data/moving_trap_data_1234.jld2")["s_values"][1]

    diffusions = []
    for ε in εs   
        m = [mol(ε, x_p - x_q) for x_q in xps, x_p in xps]
        g = [grad_mol(ε, x_p-x_q) for x_q in xps, x_p in xps]
        M = reshape(sum(m, dims=1), :)
        G = reshape(sum(g, dims=1), :)
        G_by_M = reduce(hcat, G ./ M)
        g_by_M = reduce(hcat, g * (one(eltype(M)) ./ M))
        diffusion = D(xs, t) * reshape(G_by_M + g_by_M, d_bar, N, n)

        push!(diffusions, diffusion)
    end
    # plot on log scale
    pl1 = plot(εs, norm.(diffusions, 1), label = "jhu diffusion", xlabel = "ε", ylabel = "1-norm", title = "N = $N, diffusion vs drift norms at update step", yscale = :log10, xticks = εs)
    plot!(pl1, εs, repeat([norm(D(xs,t)*s_value, 1)], length(εs)), label = "sbtm diffusion")
    plot!(pl1, εs, repeat([norm(drift, 1)], length(εs)), label = "drift")
    # pl2 = plot(εs, norm.(diffusions, Inf)./N, label = nothing, xscale = :log10, yscale = :log10, xlabel = "ε", ylabel = "Inf-norm/N")
    # plot!(pl2, εs, repeat([norm(s_value, Inf)/N], length(εs)), label = nothing)
    # plot!(pl2, εs, repeat([norm(drift, Inf)/N], length(εs)), label = nothing)
end

print_mol_stats(N, num_samples, num_timestamps)


# stats_plot, heatmap_plot = sbtm_vs_jhu_experiment(N, num_samples, num_timestamps)