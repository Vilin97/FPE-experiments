using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux, Roots
using Random: seed!

include("blob.jl")
include("sbtm.jl")
include("../utils.jl")
include("../plotting_utils.jl")

const START = 5.5
const SAVEAT = [0., 0.25, 0.5]

a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+START)/6)
# true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * pdf(MvNormal(K*I(3)), x)
true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * (2π*K)^(-3/2) * exp(-sum(abs2, x)/(2K))
true_marginal(x :: Number, K) = (a(K) + b(K)*x^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), [x])
true_marginal(x :: AbstractVector, K) = (a(K) + b(K)*x[1]^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), x)
true_slice(x, K) = true_pdf([x,0,0], K)

"plot the slice pdfs at time t"
function pdf_plot(solutions, labels, ε, time_index; num_runs=1)
    t = SAVEAT[time_index]
    n = num_particles(solutions[1][1]) ÷ num_runs
    plt = plot(title = "t = $(round(START+t, digits = 2)), n = $n", xlabel = "x", ylabel = "pdf(x)", ylim = (0, 0.05))
    pdf_range = range(-6, 6, length=100)
    label_ = "true"
    plot!(plt, pdf_range, x->true_slice(x, K(t)), label = label_)
    for (solution, label) in zip(solutions, labels)
        u = solution[time_index]
        plot!(plt, pdf_range, x -> reconstruct_pdf(ε, [x,0,0], u), label = "$label, ε = $(round(ε, digits = 4))")
    end
    plt
end

"Change n, plot slice pdfs at end time."
function slice_pdf_experiment()
    num_runs = 5
    time_index = 3
    ns = [200, 400, 1000, 2000, 4000, 10_000]
    # ns = [200, 500]
    plots = []
    for n in ns
        ε = 1 / n^(1/3)
        solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$num_runs.jld2", "solution")
        solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_runs_$num_runs.jld2", "solution")
        pdf_plt = pdf_plot([solution_sbtm, solution_blob], ["sbtm", "blob"], ε, time_index, num_runs=num_runs)
        push!(plots, pdf_plt)
    end
    plot(plots..., layout = (2,3), size = (1600, 900))
end

plt = slice_pdf_experiment()
savefig(plt, "plots/landau 3d slice pdfs combined")

"Change n, plot Lp error of k-particles marginal at end time vs n."
function Lp_error_experiment(d=3; p=2, k=3, verbose = 0, kwargs...)
    ns = [50, 100, 200, 400, 500, 1000, 2000, 4000, 10_000]
    t_range = SAVEAT
    num_runs = 5
    plots = []
    for (i,t) in enumerate(t_range)
        sbtm_errors = Float64[]
        blob_errors = Float64[]
        @views for n in ns
            ε = 1 / n^(1/3)
            solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$num_runs.jld2", "solution")
            solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_runs_$num_runs.jld2", "solution")
            lp_error_sbtm = Lp_error_slice(solution_sbtm[i], x->true_pdf(x, K(t)); reconstruct_ε = ε, p=p, k=k, verbose = verbose, kwargs...)
            lp_error_blob = Lp_error_slice(solution_blob[i], x->true_pdf(x, K(t)); reconstruct_ε = ε, p=p, k=k, verbose = verbose, kwargs...)
            push!(sbtm_errors, lp_error_sbtm)
            push!(blob_errors, lp_error_blob)
        end
        Lp_errors_plot = Lp_error_plot(ns, [sbtm_errors, blob_errors], ["sbtm", "blob"], [:red, :green], t, d, "landau", k, p)
        push!(plots, Lp_errors_plot)
    end
    Lp_plots = plot(plots..., layout = (length(plots), 1), size = (1000, 1000))
end

L2_errror_plot = Lp_error_experiment(;k=3, verbose = 1, max_evals=3*10^3, xlim = 5)
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