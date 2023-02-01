using Test, Distributions, Random, TimerOutputs

include("../src/utils.jl")
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
    seed!(1234)
    reconstruct_pdf(ε, x, u) = Mol(ε, x, u)/size(u, 2)
    function L2_norm(f, g, h, lims) # h is mesh size, lims is the bounds of the domain
        res = 0.
        for x1 in lims[1]:h:lims[2], x2 in lims[1]:h:lims[2]
            x = [x1, x2]
            res += (f(x) - g(x))^2 * h^2
        end
        sqrt(res)
    end

    function L2_error(solution, ε)
        reconstructed = x -> reconstruct_pdf(ε, x, solution[end])
        analytic = x -> pdf(ρ(solution.t[end]), x)
        L2_norm(reconstructed, analytic, 0.1, (-3, 3))
    end

    D(x,t) = 1.
    b(x,t) = zero(x)

    ε = 0.05
    tspan = (0., 0.5)
    dt = 0.01
    ts = tspan[1]:dt:tspan[2]
    Δts = repeat([dt], length(ts)-1)
    d = 2
    n = 5000
    ρ(t) = MvNormal(2*(t+1)*I(d))
    xs = reshape(rand(ρ(0.), n), d, 1, n)
    rtol = 1.

    @timeit "jhu" trajectory, solution = jhu(xs, Δts, b, D; ε = ε)
    l2_error = L2_error(solution, ε)
    print("L2 error for jhu is $l2_error")
    @timeit "L2_error_test_jhu" @test l2_error < rtol

    @timeit "sbtm" trajectory, extras = sbtm(xs, Δts, b, D; ρ₀ = ρ(0.), verbose = 0)
    solution = extras["solution"]
    @test !isnothing(solution)
    l2_error = L2_error(solution, ε)
    print("L2 error for sbtm is $l2_error")
    @timeit "L2_error_test_sbtm" @test l2_error < rtol
end

TimerOutputs.reset_timer!()
@timeit "no-diffusion test" @testset "no-diffusion test" no_diffusion_test()
@timeit "diffusion test" @testset "diffusion test" diffusion_test()
TimerOutputs.print_timer()
