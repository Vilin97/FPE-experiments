using Test, Distributions, Random, TimerOutputs

include("../src/utils.jl")
include("../src/fpe/blob.jl")
include("../src/fpe/sbtm.jl")
include("../src/fpe/exact.jl")

function no_diffusion_test()
    println("Testing FPE pure drift")
    Random.seed!(1234)
    b(x,t) = x
    D(x,t) = 0.
    x0 = 1.0
    ρ(t) = x0 * exp(t)
    dt = 0.01
    tspan = (0.0, 1.0)
    ts = tspan[1]:dt:tspan[2]
    xs = reshape([x0], 1, 1)
    rtol = 0.01

    @timeit "blob" solution = blob_fpe(xs, ts, b, D; ε = 1/π, saveat = ts)
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol

    # the NN does not matter here because D is zero
    @timeit "sbtm" solution, _, _ = sbtm_fpe(xs, ts, b, D; s = Dense(1 => 1), saveat = ts)
    @test first.(solution.u) ≈ [ρ(t) for t in solution.t] rtol = rtol
end
function diffusion_test()
    println("Testing FPE pure diffusion")
    Random.seed!(1235)
    
    ε = 0.05
    d = 2
    n = 1000
    xs, ts, b, D, ρ₀, ρ = pure_diffusion(d, n; dt = 0.01, t_end = 10.)
    exact_score(t, x) = score(ρ(t), x)
    p = 1
    k = 1
    # error tolerance
    lp_tol = 0.2
    mean_tol = 0.1
    cov_tol = 15
    
    # cpu
    @timeit "blob" blob_solution = blob_fpe(xs, ts, b, D; ε = ε, saveat = ts[end])
    Random.seed!(1235)
    @timeit "sbtm" sbtm_solution, _, _ = sbtm_fpe(xs, ts, b, D; ρ₀ = ρ(0.), verbose = 0, saveat = ts[end])
    @timeit "exact" exact_solution = exact_fpe(xs, ts, b, D; exact_score = exact_score, saveat = ts[end])

    # gpu
    xsg = gpu(xs); tsg = gpu(ts); ε = Float32(ε)
    @timeit "blob" blob_solution_gpu = blob_fpe(xsg, tsg, b, D; ε = ε, saveat = ts[end])
    Random.seed!(1235)
    @timeit "sbtm" sbtm_solution_gpu, _, _ = sbtm_fpe(xsg, tsg, b, D; ρ₀ = ρ(0.), verbose = 0, saveat = ts[end])
    @test cpu(blob_solution_gpu.u) ≈ blob_solution.u rtol = 1e-2
    @test cpu(sbtm_solution_gpu.u) ≈ sbtm_solution.u rtol = 1e-2

    # statistical tests
    for (solution, label) in [(blob_solution, "blob"), (sbtm_solution, "sbtm"), (exact_solution, "exact")]
        @test eltype(solution[end]) == Float32

        error = Lp_error_marginal(solution, ρ, ts[end]; p=p, k = k)
        println("L$p error for $label is $error")
        @test error < lp_tol

        emp_mean = empirical_first_moment(solution[end])
        emp_cov = empirical_covariance(solution[end])
        cov_diff = cov(ρ(ts[end])) - emp_cov
        println("$(label) mean error = $(norm(emp_mean))")
        println("$(label) cov norm error = $(norm(cov_diff))")
        println("$(label) cov trace error = $(tr(cov_diff))")
        @test norm(emp_mean) < mean_tol
        @test norm(cov_diff) < cov_tol
    end
end

no_diffusion_test()
diffusion_test()