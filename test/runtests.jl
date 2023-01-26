using Test, Distributions, Random

include("../src/jhu.jl")
include("../src/sbtm.jl")

# no-diffusion test
function no_diffusion_test()
    seed!(1234)
    b(x,t) = x
    D(x,t) = zero(eltype(x))
    x0 = 1.0
    ρ(t) = x0 * exp(t)
    dt = 0.01
    tspan = (0.0, 1.0)
    xs = reshape([x0], 1, 1, 1)
    num_ts = Int(tspan[2]/dt)
    Δts = repeat([dt], num_ts)
    rtol = 0.1

    trajectory, solution = jhu(xs, Δts, b, D; ε = 1/π)
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol

    trajectory, extras = sbtm(xs, Δts, b, D; s = Dense(1 => 1))
    solution = extras["solution"]
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol
end

# no-drift test
# TODO