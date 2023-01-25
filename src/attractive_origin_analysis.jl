# analyze the trajectories from the first example in the paper

using DifferentialEquations, JLD2, Distributions, ComponentArrays, LinearAlgebra, Plots
using Random: seed!
include("utils.jl")
plotly()

# reconstruct the initial parameters
function initial_params(num_samples, num_timestamps; seed = num_samples*num_timestamps)
    seed!(seed)
    xs, Δts, b, D, ρ₀, ρ = attractive_origin(num_samples, num_timestamps)
    ts = vcat(0f0, cumsum(Δts))
    xs, Δts, b, D, ρ₀, ρ, ts
end

### ANALYTICAL ###
analytic_entropies(ρ, ts) = entropy.(ρ.(ts))
analytic_energies(ρ, ts) = analytic_entropies(ρ, ts) .+ ρ.(ts) .|> cov .|> det .|> √
analytic_moments(ρ, ts) = mean.(ρ.(ts)), cov.(ρ.(ts))
analytic_marginals(xs, t, ρ) = pdf(ρ(t), xs)

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

function make_plots(trajectories, labels; ε_entropy = 1.24, ε_marginal = 0.15)
    println("Making plots")
    d_bar, N, num_samples, _ = size(trajectories[1])
    xs, Δts, b, D, ρ₀, ρ, ts = initial_params(num_samples, size(trajectories[1], 4)-1)
    analytic_expectations, analytic_covariances = analytic_moments(ρ, ts)
    analytic_entropies_ = analytic_entropies(ρ, ts)

    # for heatmap plotting
    range = -3:0.1:3
    x_grid = Iterators.product(range, range)
    time_indices = 1:size(trajectories[1], 4)÷4:size(trajectories[1], 4)
    heatmaps = Matrix{Any}(undef, length(trajectories)+1, length(time_indices))
    analytic_marginals_ = [analytic_marginals(collect.(x_grid), t, ρ) for t in ts[time_indices]]

    heatmaps[1, :] = [heatmap(range, range, analytic_marginals_[i]', title = "analytic marginal at t = $(round(t, digits = 2))") for (i, t) in enumerate(ts[time_indices])]

    p1 = plot(title = "mean position comparison, num_samples = $(num_samples)", ylabel = "mean position 2-norm")
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

# sbtm vs jhu
function sbtm_vs_jhu_experiment(num_samples, num_timestamps)
    println("Loading data for num_samples = $num_samples, num_timestamps = $num_timestamps")
    seed = num_samples*num_timestamps
    seed!(seed)
    data_jhu = JLD2.load("data/attractive_origin_jhu_$seed.jld2")
    data_sbtm = JLD2.load("data/attractive_origin_sbtm_$seed.jld2")
    jhu_trajectories = data_jhu["trajectories"]
    sbtm_trajectories = data_sbtm["trajectories"]
    trajectories = [sbtm_trajectories, jhu_trajectories]
    labels = ["sbtm", "jhu"]

    stats_plot, heatmap_plot = make_plots(trajectories, labels)
end

# assume num_samples, num_timestamps are already defined
stats_plot, heatmap_plot = sbtm_vs_jhu_experiment(num_samples, num_timestamps)