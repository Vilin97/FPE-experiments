# analyze the trajectories from the first example in the paper

using DifferentialEquations

# NOTE: assumes the initial condition is (0,0)
f(m,p,t) = β(t) - m
sol = solve(ODEProblem(f, zeros(Float32, 2), (0f0, 2f0)))
plot(sol)