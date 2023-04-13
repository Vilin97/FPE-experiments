using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux, Roots
using Random: seed!

include("blob.jl")
include("sbtm.jl")
include("../utils.jl")
include("../plotting_utils.jl")

a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+6.)/6)
# true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * pdf(MvNormal(K*I(3)), x)
true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * (2π*K)^(-3/2) * exp(-sum(abs2, x)/(2K))
true_marginal(x :: Number, K) = (a(K) + b(K)*x^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), [x])
true_marginal(x :: AbstractVector, K) = (a(K) + b(K)*x[1]^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), x)

"plot the marginal pdfs at time t"
function pdf_plot(solutions, labels, ε, d = 3)
    time_index = 2
    n = num_particles(solutions[1][1])
    plt = plot(title = "t = $(round(6.5, digits = 2)), n = $n", xlabel = "x", ylabel = "pdf(x)", ylim = (0, 0.5))
    pdf_range = range(-6, 6, length=100)
    label_ = "true"
    plot!(plt, pdf_range, [true_marginal(x, K(0.5)) for x in pdf_range], label = label_)
    for (solution, label) in zip(solutions, labels)
        u = solution[time_index][1, :]
        plot!(plt, pdf_range, [reconstruct_pdf(ε, x, u) for x in pdf_range], label = label)
    end
    plt
end

"Change n, plot marginal pdfs at end time."
function marginal_pdf_experiment()
    ε = 0.053
    ns = [200, 500, 1000, 2000, 4000, 10_000]
    plots = []
    for n in ns
        # t_plot = 0.5
        time_index = 2
        solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_avg_10.jld2", "average_solution")
        solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_avg_10.jld2", "average_solution")
        pdf_plt = pdf_plot([solution_sbtm, solution_blob], ["sbtm", "blob"], time_index, ε)
        push!(plots, pdf_plt)
    end
    plot(plots..., layout = (2, 3), size = (1600, 900))
end

marginal_plot = marginal_pdf_experiment()
savefig(marginal_plot, "plots/landau 3d marginal pdfs combined")

"Change n, plot Lp error of k-particles marginal at end time vs n."
function Lp_error_experiment(d=3; p=2, k=3, verbose = 0)
    ε = 0.053
    ns = [50, 100, 200, 400, 500, 1000, 2000, 4000, 10_000]
    t_range = collect(range(0.25, 0.5, length = 2))
    plots = []
    for (i,t) in enumerate(t_range)
        sbtm_errors = Float64[]
        blob_errors = Float64[]
        for n in ns
            solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_avg_10.jld2", "average_solution")
            solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_avg_10.jld2", "average_solution")
            push!(sbtm_errors, Lp_error(solution_sbtm, ε, i, t_range, d, n; verbose = verbose, p = p, k=k, marginal_pdf = x -> true_pdf(x, K(t))))
            push!(blob_errors, Lp_error(solution_blob, ε, i, t_range, d, n; verbose = verbose, p = p, k=k, marginal_pdf = x -> true_pdf(x, K(t))))
        end
        Lp_errors_plot = Lp_error_plot(ns, [sbtm_errors, blob_errors], ["sbtm", "blob"], [:red, :green], t, d, "landau", k, p)
        push!(plots, Lp_errors_plot)
    end
    Lp_plots = plot(plots..., layout = (1, length(plots)), size = (1400, 600))
end

L2_errror_plot = Lp_error_experiment(verbose = 1)
savefig(L2_errror_plot, "plots/landau 3d L2 error combined")

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
savefig(losses_plot, "plots/landau 3d losses sbtm")








plt = plot()
for t in -0.5:0.1:0.5
    plot!(plt, -2:0.1:2, x -> true_pdf([x,0,0], K(t)), label = "t = $(6+t)", title = "slice of true pdf, ρ(x,0,0)", xlabel = "x", ylabel = "pdf(x)")
end
plt

n = 1000
solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_avg_10.jld2", "average_solution")
solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_avg_10.jld2", "average_solution")
diff = solution_blob[1] - solution_blob[2]