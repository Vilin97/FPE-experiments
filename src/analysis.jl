# analyze the trajectories from the first example in the paper

using DifferentialEquations, JLD2, Distributions, ComponentArrays

data = load("moving_trap_data.jld2")
sbtm_trajectories = data["sbtm_trajectories"]  
losses = data["losses"] 
s_values = data["s_values"] 
jhu_trajectories = data["jhu_trajectories"] 
xs, Δts, b, D, ρ₀, target, a, w, α, β = data["initial_parameters"]

tspan = (0f0, sum(Δts))
f_mean(m,p,t) = β(t) - m
sol_mean = solve(ODEProblem(f_mean, mean(ρ₀), tspan))

p = (α, size(xs, 2), size(xs, 3), t -> D(xs, t)) # assumes D is constant in space
initial = ComponentVector(Cd = cov(ρ₀), Co = zeros(eltype(xs), 2, 2))
function f_cov!(dcov, cov, p, t :: T) where T
    α, N, n, D = p
    dcov.Cd = T(2)*(α-one(T))cov[1] - T(2)*α/N*(cov[1] + (n-1)*cov[2]) + T(2)*D(t)*I
    dcov.Co = T(2)*(α-one(T))cov[2] - T(2)*α/N*(cov[1] + (n-1)*cov[2])
end
sol_cov = solve(ODEProblem(f_cov!, initial, tspan, p))