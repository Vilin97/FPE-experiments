using Test, Distributions, Random, TimerOutputs

include("../src/utils.jl")
include("../src/jhu.jl")
include("../src/sbtm.jl")

# TODO refactor:
# take ts instead of \Delta ts as input
# output only solution
# operate on (d, n) arrays instead of (d_bar, N, n) arrays. Or at least unify the dimension of output of jhu and sbtm

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
    Random.seed!(1234)
    reconstruct_pdf(ε, x, u :: AbstractMatrix) = Mol(ε, x, u)/size(u, 2)

    function L2_error(solution, true_solution, ε, t, d, n; h = 0.1)
        pdf_range = [[x,y] for x in -3:h:3, y in -3:h:3]
        u = reshape(solution(t), d, n)
        pdf_diff = [reconstruct_pdf(ε, x, u) for x in pdf_range] .- [pdf(true_solution(t), x) for x in pdf_range]
        l2_error = norm(pdf_diff) * sqrt(h^d)
    end

    D(x,t) = 1.
    b(x,t) = zero(x)

    ε = 0.05
    tspan = (0., 0.5)
    dt = 0.01
    ts = tspan[1]:dt:tspan[2]
    Δts = repeat([dt], length(ts)-1)
    d = 2
    n = 100
    ρ(t) = MvNormal(2*(t+1)*I(d))
    xs = reshape(rand(ρ(0.), n), d, 1, n)
    tol = 1.

    @timeit "jhu" trajectory, solution = jhu(xs, Δts, b, D; ε = ε)
    @test !isnothing(solution)
    l2_error = L2_error(solution, ρ, ε, tspan[2], d, n)
    println("L2 error for jhu is $l2_error")
    @timeit "L2_error_test_jhu" @test l2_error < tol

    @timeit "sbtm" trajectory, extras = sbtm(xs, Δts, b, D; ρ₀ = ρ(0.), verbose = 0)
    solution = extras["solution"]
    @test !isnothing(solution)
    l2_error = L2_error(solution, ρ, ε, tspan[2], d, n)
    println("L2 error for sbtm is $l2_error")
    @timeit "L2_error_test_sbtm" @test l2_error < tol
end

TimerOutputs.reset_timer!()
@timeit "no-diffusion test" @testset "no-diffusion test" no_diffusion_test()
@timeit "diffusion test" @testset "diffusion test" diffusion_test()
TimerOutputs.print_timer()
