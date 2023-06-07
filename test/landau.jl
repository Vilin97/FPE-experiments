using Plots, Test
using Random: seed!
include("../src/utils.jl")
include("../src/landau/blob.jl")
include("../src/landau/sbtm.jl")
include("../src/landau/exact.jl")

################### sampling ####################
function test_sampling()
    t_start = 5.5
    a(K) = (5K-3)/(2K)
    b(K) = (1-K)/(2K^2)
    K(t) = 1 - exp(-(t+t_start)/6)
    f(x, K) = pdf(MvNormal(K * I(3)), x) * (a(K) + b(K)*norm(x)^2)
    analytic_marginal(x, K) = (a(K) + b(K)*x^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), [x])


    n = 100_000
    seed!(1234)
    xs, ts, _ = landau(n, t_start)

    # marginal
    x1s = xs[1, :]
    plt_marginal = histogram(x1s, normed=true, label="sampled");
    plot!(plt_marginal, x -> analytic_marginal(x,K(0.)), label="analytic");
    plot!(plt_marginal, title = "Rejection sampling for Landau", ylabel = "marginal density", xlabel = "x")

    # slice
    x_range = -4:0.01:4
    ε = 0.025
    plt_slice = plot(title = "slice comparison, n = $n, ε = $ε", ylabel = "density", xlabel = "x");
    plot!(plt_slice, x_range, x -> f([x,0,0], K(0)), label="analytic");
    plot!(plt_slice, x_range, x -> reconstruct_pdf(ε, [x,0,0], xs), label="sampled")

    plot(plt_marginal, plt_slice, layout = (2,1), size = (1200, 600))
end

##################### solving ####################

function landau_test(verbose = 1)
    println("Testing Landau")
    seed!(1234)
    
    # t_start = 5.5
    a(K) = (5K-3)/(2K)
    b(K) = (1-K)/(2K^2)
    K(t) = 1 - exp(-(t+5.5)/6)

    reconstruction_ε = 0.1
    time_interval = 0.5
    ε = 0.05
    n = 2_000
    tol = 0.05
    p = 2

    xs, ts, f = landau(n, 5.5; time_interval = time_interval)
    exact_score(t, x) = score(x_ -> f(x_, K(t)), x)
    true_cov = I(3)
    
    # error tolerance
    lp_tol = 0.2
    mean_tol = 0.1
    cov_tol = 15
    
    # solve
    @timeit "exact" exact_solution = exact_landau(xs, ts; exact_score = exact_score, saveat = ts[end])
    @timeit "blob" blob_solution = blob_landau(xs, ts; ε = ε, saveat = ts[end], verbose = 0)
    @timeit "sbtm" sbtm_solution, _, _ = sbtm_landau(xs, ts; ρ₀ = x->f(x, K(0)), saveat = ts[end], verbose = 0, loss_tolerance = 1e-3)

    # test
    initial_error = Lp_error(xs, x -> f(x, K(0)); k=3, verbose = 0)
    verbose > 0 && println("L$p error for initial sample is $initial_error.")
    @test initial_error < tol
    for (solution, label) in [(blob_solution, "blob"), (sbtm_solution, "sbtm"), (exact_solution, "exact")]
        @test eltype(solution[end]) == Float32

        error = Lp_error(solution[end], x -> f(x, K(time_interval)); k=3, verbose = 0)
        println("L$p error for $label is $error.")
        @test error < lp_tol

        emp_mean = empirical_first_moment(solution[end])
        emp_cov = empirical_covariance(solution[end])
        cov_diff = true_cov .- emp_cov
        verbose > 0 && println("$(label) mean error = $(norm(emp_mean))")
        verbose > 0 && println("$(label) cov norm error = $(norm(cov_diff))")
        verbose > 0 && println("$(label) cov trace error = $(tr(cov_diff))")
        @test norm(emp_mean) < mean_tol
        @test norm(cov_diff) < cov_tol
    end
end

landau_test()