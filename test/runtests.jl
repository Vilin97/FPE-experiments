using Test, Distributions, Random, TimerOutputs

include("../src/jhu.jl")
include("../src/sbtm.jl")

# no-diffusion test
function no_diffusion_test()
    Random.seed!(1234)
    b(x,t) = x
    D(x,t) = 0.
    x0 = 1.0
    ρ(t) = x0 * exp(t)
    dt = 0.01
    tspan = (0.0, 1.0)
    xs = reshape([x0], 1, 1, 1)
    num_ts = Int(tspan[2]/dt)
    Δts = repeat([dt], num_ts)
    rtol = 0.1

    jhu(xs, Δts, b, D; ε = 1/π)
    @timeit "jhu" trajectory, solution = jhu(xs, Δts, b, D; ε = 1/π)
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol

    # the NN does not matter here because D is zero
    sbtm(xs, Δts, b, D; s = Dense(1 => 1))
    @timeit "sbtm" trajectory, extras = sbtm(xs, Δts, b, D; s = Dense(1 => 1))
    solution = extras["solution"]
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol
end

function diffusion_test()
    D(x,t) = 1.
    b(x,t) = zero(x)

    ε = 0.05
    tspan = (0., 1.0)
    dt = 0.01
    ts = tspan[1]:dt:tspan[2]
    Δts = repeat([dt], length(ts)-1)
    d = 2
    n = 100
    xs = randn(d, 1, n)

    jhu(xs, Δts, b, D; ε = ε)
    @timeit "jhu" trajectory, solution = jhu(xs, Δts, b, D; ε = ε)
    @test !isnothing(solution)

    # the NN does not matter here because we are not checking correctness (yet)
    sbtm(xs, Δts, b, D; s = Dense(d => d))
    @timeit "sbtm" trajectory, extras = sbtm(xs, Δts, b, D; s = Dense(d => d))
    solution = extras["solution"]
    @test !isnothing(solution)
end

TimerOutputs.reset_timer!()
@timeit "no-diffusion test" no_diffusion_test()
@timeit "diffusion test" diffusion_test()
TimerOutputs.print_timer()
