# analyze the trajectories from the first example in the paper

using DifferentialEquations, JLD2, Distributions, ComponentArrays, LinearAlgebra, Plots
using Random: seed!
include("utils.jl")

println("Loading data")
data = JLD2.load("moving_trap_data_1234.jld2")
sbtm_trajectories = data["sbtm_trajectories"]  
losses = data["losses"] 
s_values = data["s_values"] 
jhu_trajectories = data["jhu_trajectories"] 
@assert size(sbtm_trajectories) == size(jhu_trajectories)

# reconstruct the initial parameters
seed!(data["seed"])
d_bar, N, num_samples, num_timestamps = size(sbtm_trajectories)
xs, Δts, b, D, ρ₀, target, a, w, α, β = moving_trap(N, num_samples, num_timestamps-1)
ts = vcat(0f0, cumsum(Δts))

# solve analytically
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

function make_plots()
    sbtm_expectations, analytic_expectations, sbtm_expectation_errors, sbtm_covariances, analytic_covariances, sbtm_covariance_errors  = moment_analysis(sbtm_trajectories)
    jhu_expectations, _, jhu_expectation_errors, jhu_covariances, _, jhu_covariance_errors = moment_analysis(jhu_trajectories)

    p1 = plot(ts, sbtm_expectation_errors, label = "sbtm expectation errors")
    plot!(p1, ts, jhu_expectation_errors, label = "jhu expectation errors")
    p2 = plot(ts, sbtm_covariance_errors, label = "sbtm covariance errors")
    plot!(p2, ts, jhu_covariance_errors, label = "jhu covariance errors")
    # p1 = plot(ts, analytic_expectations, label = "analytic expectations")
    # plot!(p1, ts, sbtm_expectations, label = "sbtm expectations")
    # plot!(p1, ts, jhu_expectations, label = "jhu expectations")

    # p2 = plot(ts, jhu_expectation_errors, label = "expectation errors")
    # p2 = plot!(ts, analytic_covariances, label = "analytic covariances")
    # plot!(p2, ts, sbtm_covariances, label = "sbtm covariances")
    # plot!(p2, ts, jhu_covariances, label = "jhu covariances")
    println("Combining plots")
    plot(p1, p2)
end

plotly()
println("Making plots")
make_plots()

#TODO: compare empirical and analytic covariances for both methods
#TODO: add contour plots for the analytic probability dist to the animation
#TODO: compute Fisher divergence between sₜ and ρₜ