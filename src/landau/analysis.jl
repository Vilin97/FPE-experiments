using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux, Roots
using Random: seed!

include("../utils.jl")
include("../plotting_utils.jl")

a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+5.5)/6)
# true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * pdf(MvNormal(K*I(3)), x)
true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * (2π*K)^(-3/2) * exp(-sum(abs2, x)/(2K))
true_marginal(x :: Number, K) = (a(K) + b(K)*x^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), [x])
true_marginal(x :: AbstractVector, K) = (a(K) + b(K)*x[1]^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), x)

"plot the marginal pdfs at time t"
function pdf_plot(solutions, labels, t, ε, d = 3)
    u = reshape(solutions[1](t), d, :)
    n = size(u, 2)
    plt = plot(title = "t = $(round(5.5+t, digits = 2)), n = $n", xlabel = "x", ylabel = "pdf(x)", ylim = (0, 0.5))
    pdf_range = range(-6, 6, length=100)
    label_ = (n==200) ? "true" : nothing
    plot!(plt, pdf_range, [true_marginal(x, K(t)) for x in pdf_range], label = label_)
    for (solution, label) in zip(solutions, labels)
        u = reshape(solution(t), d, n)[1, :]
        label_ = (n==200) ? label : nothing
        plot!(plt, pdf_range, [reconstruct_pdf(ε, x, u) for x in pdf_range], label = label_)
    end
    plt
end

"Change n, plot marginal pdfs at end time."
function marginal_pdf_experiment()
    ε = 0.053
    ns = [1000, 2000, 4000, 8000, 10_000, 20_000]
    plots = []
    for n in ns
        t_plot = 0.5
        solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n).jld2", "solution")
        pdf_plt = pdf_plot([solution_sbtm], ["sbtm"], t_plot, ε)
        push!(plots, pdf_plt)
    end
    plot(plots..., layout = (2, 3), size = (1400, 800))
end

marginal_plot = marginal_pdf_experiment()
savefig(marginal_plot, "plots/landau 3d marginal pdfs sbtm")

"Change n, plot Lp error of k-particles marginal at end time vs n."
function Lp_error_experiment(d=3; p=2, k=1, verbose = 0)
    ε = 0.053
    ns = [100, 200, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10_000, 20_000]
    t_range = range(0.25, 0.5, length = 2)
    plots = []
    for (i,t) in enumerate(t_range)
        sbtm_errors = Float64[]
        for n in ns
            solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n).jld2", "solution")
            push!(sbtm_errors, Lp_error(solution_sbtm, ε, t, d, n; verbose = verbose, p = p, k=k, marginal_pdf = x -> true_marginal(x, K(t))))
        end
        Lp_errors_plot = Lp_error_plot(ns, [sbtm_errors], ["sbtm"], [:red], t, d, "landau", k, p)
        push!(plots, Lp_errors_plot)
    end
    Lp_plots = plot(plots..., layout = (1, length(plots)), size = (1400, 600))
end

L2_errror_plot = Lp_error_experiment()
savefig(L2_errror_plot, "plots/landau 3d L2 error sbtm")

function losses_experiment(n)
    losses = JLD2.load("landau_experiment/sbtm_n_$(n).jld2", "losses")
    loss_plot = plot_losses(losses)

    solution = JLD2.load("landau_experiment/sbtm_n_$(n).jld2", "solution")
    s_values = JLD2.load("landau_experiment/sbtm_n_$(n).jld2", "s_values")
    ts = solution.t
    n = num_particles(solution(0))
    errors = zeros(length(ts))
    @views for (k,t) in enumerate(ts)
        xs = solution(t)
        ys = score(x -> true_pdf(x, K(t)), xs)
        errors[k] = sum(abs2, s_values[:,:,k] - ys)/n
    end
    error_plot = plot(ts, errors, title = "1/n ∑ᵢ|s(Xᵢ) - ∇log ρ(Xᵢ)|^2", xlabel = "time", ylabel = "error", label = "n = $n")
    plot(loss_plot, error_plot, layout = (1, 2), size = (1400, 600))
end

losses_plot1 = losses_experiment(8000);
losses_plot2 = losses_experiment(20_000);
losses_plot = plot(losses_plot1, losses_plot2, layout = (2, 1), size = (1400, 1200))