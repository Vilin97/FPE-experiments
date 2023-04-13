using Test, Distributions, Random, TimerOutputs

include("../src/utils.jl")
include("../src/blob.jl")
include("../src/sbtm.jl")

function no_diffusion_test()
    Random.seed!(1234)
    b(x,t) = x
    D(x,t) = 0.
    x0 = 1.0
    ρ(t) = x0 * exp(t)
    dt = 0.01
    tspan = (0.0, 1.0)
    ts = tspan[1]:dt:tspan[2]
    xs = reshape([x0], 1, 1, 1)
    rtol = 0.1

    @timeit "blob" solution = blob(xs, ts, b, D; ε = 1/π)
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol

    # the NN does not matter here because D is zero
    @timeit "sbtm" solution = sbtm(xs, ts, b, D; s = Dense(1 => 1))
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol
end
function diffusion_test()
    Random.seed!(1234)
    
    ε = 0.05
    d = 2
    n = 100
    xs, ts, b, D, ρ₀, ρ = pure_diffusion(d, n, 0.01)
    tol = 0.01
    p = 1
    k = 1

    blob(xs, ts, b, D; ε = ε)
    @timeit "blob" solution = blob(xs, ts, b, D; ε = ε)
    @test !isnothing(solution)
    error = Lp_error(solution, ρ, ε, ts[end], d; p=p, k = k)
    println("L$p error for blob is $error")
    @timeit "blob error test" @test error < tol

    @timeit "sbtm" solution = sbtm(xs, ts, b, D; ρ₀ = ρ(0.), verbose = 0)
    @test !isnothing(solution)
    error = Lp_error(solution, ρ, ε, ts[end], d; p=p, k = k)
    println("L$p error for sbtm is $error")
    @timeit "sbtm error test" @test error < tol
end

TimerOutputs.reset_timer!()
@timeit "diffusion test" @testset "diffusion test" diffusion_test()
@timeit "no-diffusion test" @testset "no-diffusion test" no_diffusion_test()
TimerOutputs.print_timer()
