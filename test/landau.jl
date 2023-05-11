using Plots, Test
using Random: seed!
include("../src/utils.jl")
include("../src/landau/blob.jl")
include("../src/landau/sbtm.jl")

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

function landau_test()
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
    initial_error = Lp_error(xs, x -> f(x, K(0)); k=3, verbose = 1)
    println("L$p error for initial sample is $initial_error.")
    @test initial_error < tol

    @timeit "blob" blob_solution = blob_landau(xs, ts; ε = ε, saveat = ts[[1, end]], verbose = 1)
    error = Lp_error(blob_solution[end], x -> f(x, K(time_interval)); k=3, verbose = 1)
    println("L$p error for blob is $error.")
    @test error < tol

    @timeit "sbtm" sbtm_solution, _, _ = sbtm_landau(Float32.(xs), ts; ρ₀ = x->f(x, K(0)), saveat = ts[[1, end]], verbose = 1, loss_tolerance = 1e-3)
    error = Lp_error(sbtm_solution[end], x -> f(x, K(time_interval)); k=3, verbose = 1)
    println("L$p error for sbtm is $error.")
    @test error < tol
end

landau_test()