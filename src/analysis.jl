# analyze the trajectories from the first example in the paper

using DifferentialEquations, JLD2, Distributions, ComponentArrays, LinearAlgebra, Plots
using Random: seed!
include("utils.jl")
plotly()

println("Loading data")
data = JLD2.load("data/moving_trap_data_1234.jld2")
sbtm_trajectories = data["sbtm_trajectories"]  
jhu_trajectories = data["jhu_trajectories"] 
@assert size(sbtm_trajectories) == size(jhu_trajectories)

# reconstruct the initial parameters
seed!(data["seed"])
d_bar, N, num_samples, num_timestamps = size(sbtm_trajectories)
xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps-1)
ts = vcat(0f0, cumsum(Δts))

# solve analyticly
tspan = (0f0, sum(Δts))
p = (α, size(xs, 2), t -> D(xs, t), β) # assumes D is constant in space
initial = ComponentVector(m = mean(ρ₀), Cd = cov(ρ₀), Co = zeros(eltype(xs), 2, 2))
function f!(dcov, cov, p, t :: T) where T
    α, N, D, β = p
    dcov.m = β(t) - cov.m
    dcov.Cd = T(2)*(α-one(T))*cov.Cd - T(2)*α/N*(cov.Cd + (N-1)*cov.Co) + T(2)*D(t)*I
    dcov.Co = T(2)*(α-one(T))*cov.Co - T(2)*α/N*(cov.Cd + (N-1)*cov.Co)
end
analytic_solution = solve(ODEProblem(f!, initial, tspan, p), saveat = ts)
ρ(t) = MvNormal(kron(ones(N), analytic_solution(t).m), kron((analytic_solution(t).Cd - analytic_solution(t).Co), I(N)) + kron(analytic_solution(t).Co, ones(Float32, N, N)))

function analytic_score(analytic_solution, time_index, x)
    # NOTE: assumes the initial covariance matrix is scalar*I
    u = analytic_solution[time_index]
    @assert u.Cd[1,1] == u.Cd[2,2]
    @assert isdiag(u.Cd)
    @assert isdiag(u.Co)
    σ² = u.Cd[1,1] # variance
    m = u.m
    return σ² .* (x .- m)
end

function analytic_entropy(t)
    d, = size(ρ(t))
    return d/2*(log(2*π) + 1) + 1/2*log(det(cov(ρ(t))))
end

function empirical_entropy(ε, flat_xs :: AbstractArray{T, 2}) where T
    d, n = size(flat_xs)
    return -sum( log(Mol(ε, x, flat_xs)/n) for x in eachslice(flat_xs, dims=2) )/n
end

"empirical marginal pdf at point x ∈ R²"
function empirical_marginal(ε, x, xs :: AbstractArray{T, 3}) where T
    Mol(ε, x, xs[:,1,:])/size(xs, 3)
end

"analytic marginal pdf at point x ∈ R²"
function analytic_marginal(x, t :: Real)
    d_bar = length(x)
    sol = ρ(t)
    pdf(MvNormal(mean(sol)[1:d_bar], cov(sol)[1:d_bar, 1:d_bar]), x)
end

function analytic_marginals(xs, time_index, analytic_solution)
    d_bar = length(first(xs))
    sol = analytic_solution[time_index]
    marginal_dist = MvNormal(sol.m, I(d_bar)*(sol.Cd - sol.Co)[1] + ones(Float32, d_bar, d_bar)*sol.Co[1])
    pdf(marginal_dist, xs)
end

# empirical statistics
empirical_first_moment(flat_xs :: AbstractArray{T, 2}) where T = vec(mean(flat_xs, dims = 2))
empirical_second_moment(flat_xs :: AbstractArray{T, 2}) where T = flat_xs * flat_xs' / (size(flat_xs, 2) - 1)
empirical_covariance(flat_xs :: AbstractArray{T, 2}) where T = empirical_second_moment(flat_xs .- empirical_first_moment(flat_xs))

function moment_analysis(trajectories)
    flat_trajectories = reshape(trajectories, d_bar*N, num_samples, num_timestamps)
    expectations = [empirical_first_moment(flat_xs) for flat_xs in eachslice(flat_trajectories, dims=3)]
    covariances = [empirical_covariance(flat_xs) for flat_xs in eachslice(flat_trajectories, dims=3)]
    analytic_expectations = [mean(ρ(t)) for t in ts]
    analytic_covariances = [cov(ρ(t)) for t in ts]
    expectation_errors = [norm(expectation - analytic_expectation) for (expectation, analytic_expectation) in zip(expectations, analytic_expectations)]
    covariance_errors = [tr(cov - analytic_cov) for (cov, analytic_cov) in zip(covariances, analytic_covariances)]
    expectations, analytic_expectations, expectation_errors, covariances, analytic_covariances, covariance_errors
end

function entropy_analysis(trajectories, ε)
    flat_trajectories = reshape(trajectories, d_bar*N, num_samples, num_timestamps)
    entropies = [empirical_entropy(ε, flat_xs) for flat_xs in eachslice(flat_trajectories, dims=3)]
    analytic_entropies = [analytic_entropy(t) for t in ts]
    entropies, analytic_entropies
end

function make_plots(trajectories, labels; ε_entropy = 1.24, ε_marginal = 0.15)
    println("Making plots")
    _, analytic_expectations, _, _, analytic_covariances, _  = moment_analysis(trajectories[1])
    _, analytic_entropies = entropy_analysis(trajectories[1], ε_entropy)

    # for heatmap plotting
    range = -3:0.1:3
    x_grid = Iterators.product(range, range)
    time_indices = 1:50:201
    heatmaps = Matrix{Any}(undef, length(trajectories)+1, length(time_indices))
    analytic_marginals_ = [analytic_marginals(collect.(x_grid), time_index, analytic_solution) for time_index in time_indices]

    heatmaps[1, :] = [heatmap(range, range, analytic_marginals_[i]', title = "analytic marginal at t = $(round(t, digits = 2))") for (i, t) in enumerate(ts[time_indices])]

    p1 = plot(title = "expectation comparison", ylabel = "mean position 2-norm")
    plot!(p1, ts, norm.(analytic_expectations), label = "analytic")

    p2 = plot(title = "covariance trace comparison", ylabel = "covariance trace")
    plot!(p2, ts, tr.(analytic_covariances), label = nothing)

    p3 = plot(title = "covariance norm comparison", ylabel = "covariance 1-norm")
    plot!(p3, ts, norm.(analytic_covariances, 1), label = nothing)

    p4 = plot(title = "entropy comparison", ylabel = "entropy")
    plot!(p4, ts, analytic_entropies, label = nothing)

    for (j, trajectory) in enumerate(trajectories)
        expectations, _, _, covariances, analytic_covariances, _ = moment_analysis(trajectory)
        entropies, _ = entropy_analysis(trajectory, ε_entropy)

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
    stats_plot, heatmap_plot = make_plots(trajectories, labels; ε_marginal = 0.14)
    stats_plot, heatmap_plot
end

# sbtm vs jhu
function sbtm_vs_jhu_experiment()
    trajectories = [sbtm_trajectories, jhu_trajectories]
    labels = ["sbtm", "jhu"]
    stats_plot, heatmap_plot = make_plots(trajectories, labels; ε_marginal = 0.14)
    stats_plot, heatmap_plot
end

# no_drift_experiment
function no_drift_experiment()
    data = JLD2.load("no_drift_experiment/moving_trap_data_$(1000000).jld2")
    sbtm_trajectories = data["sbtm_trajectories"]  
    jhu_trajectories = data["jhu_trajectories"] 
    trajectories = [sbtm_trajectories, jhu_trajectories]
    labels = ["sbtm", "jhu"]
    stats_plot, heatmap_plot = make_plots(trajectories, labels; ε_marginal = 0.14)
    stats_plot, heatmap_plot
end

stats_plot, heatmap_plot = no_drift_experiment()