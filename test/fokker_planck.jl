using Test, Distributions, Random, TimerOutputs

include("../src/utils.jl")
include("../src/fpe/blob.jl")
include("../src/fpe/sbtm.jl")

function no_diffusion_test()
    println("Testing FPE deterministic")
    Random.seed!(1234)
    b(x,t) = x
    D(x,t) = 0.
    x0 = 1.0
    ρ(t) = x0 * exp(t)
    dt = 0.01
    tspan = (0.0, 1.0)
    ts = tspan[1]:dt:tspan[2]
    xs = reshape([x0], 1, 1, 1)
    rtol = 0.01

    @timeit "blob" solution = blob_fpe(xs, ts, b, D; ε = 1/π, saveat = ts)
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol

    # the NN does not matter here because D is zero
    @timeit "sbtm" solution, _, _ = sbtm_fpe(xs, ts, b, D; s = Dense(1 => 1), saveat = ts)
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol
end
function diffusion_test()
    println("Testing FPE pure diffusion")
    Random.seed!(1234)
    
    ε = 0.05
    d = 2
    n = 1000
    xs, ts, b, D, ρ₀, ρ = pure_diffusion(d, n, 0.01, 10.)
    p = 1
    k = 1
    # error tolerance
    lp_tol = 0.2
    mean_tol = 0.1
    cov_tol = 15

    @timeit "blob" blob_solution = blob_fpe(xs, ts, b, D; ε = ε, saveat = ts[end])
    @timeit "sbtm" sbtm_solution, _, _ = sbtm_fpe(xs, ts, b, D; ρ₀ = ρ(0.), verbose = 0, saveat = ts[end])
    for (solution, label) in [(blob_solution, "blob"), (sbtm_solution, "sbtm")]
        error = Lp_error_marginal(solution, ρ, ts[end]; p=p, k = k)
        println("L$p error for $label is $error")
        @test error < lp_tol

        xs = reshape(solution[end], d, n)
        emp_mean = empirical_first_moment(xs)
        emp_cov = empirical_covariance(xs)
        cov_diff = cov(f(ts[end])) - emp_cov
        println("$(label) mean error = $(norm(emp_mean))")
        println("$(label) cov norm error = $(norm(cov_diff))")
        println("$(label) cov trace error = $(tr(cov_diff))")
        @test norm(emp_mean) < mean_tol
        @test norm(cov_diff) < cov_tol
    end
end

@timeit "no-diffusion test" @testset "no-diffusion test" no_diffusion_test()
@timeit "diffusion test" @testset "diffusion test" diffusion_test()