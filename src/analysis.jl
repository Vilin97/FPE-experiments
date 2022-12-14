# analyze the trajectories from the first example in the paper

using DifferentialEquations, JLD2, Distributions, ComponentArrays

data = load("moving_trap_data_1234.jld2")
sbtm_trajectories = data["sbtm_trajectories"]  
losses = data["losses"] 
s_values = data["s_values"] 
jhu_trajectories = data["jhu_trajectories"] 
xs, Δts, b, D, ρ₀, target, a, w, α, β = data["initial_parameters"]

tspan = (0f0, sum(Δts))
p = (α, size(xs, 2), t -> D(xs, t), β) # assumes D is constant in space
initial = ComponentVector(m = mean(ρ₀), Cd = cov(ρ₀), Co = zeros(eltype(xs), 2, 2))
function f!(dcov, cov, p, t :: T) where T
    α, N, D, β = p
    dcov.m = β(t) - cov.m
    dcov.Cd = T(2)*(α-one(T))*cov.Cd - T(2)*α/N*(cov.Cd + (N-1)*cov.Co) + T(2)*D(t)*I
    dcov.Co = T(2)*(α-one(T))*cov.Co - T(2)*α/N*(cov.Cd + (N-1)*cov.Co)
end
analytic_solution = solve(ODEProblem(f!, initial, tspan, p), saveat = vcat(0f0, cumsum(Δts)))

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

empirical_first_moment(xs) = mean(eachslice(xs, dims = 3))
function empirical_second_moment(xs :: AbstractArray{T, 3}) where T
    d_bar, N, n = size(xs)
    result = zeros(T, d_bar, d_bar, N, N)
    for k in 1:n
        for j in 1:N
            for i in 1:N
                result[:,:,i,j] += xs[:,i,k] * xs[:,j,k]'
            end
        end
    end
    result ./ n
end

#TODO: fix second moment
#TODO: compare empirical and analytic covariances for both methods
#TODO: add contour plots for the analytic probability dist to the animation
#TODO: compute Fisher divergence between sₜ and ρₜ