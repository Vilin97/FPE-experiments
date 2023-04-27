using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux, Roots
using Random: seed!

include("blob.jl")
include("sbtm.jl")
include("../utils.jl")
include("../plotting_utils.jl")

const START = 6
const SAVEAT = [0., 0.75, 1.5]
const NUMRUNS = 10
const ns = [500, 1_000, 2_000, 4_000, 10_000, 15_000]
path(n, num_runs, start) = "landau_experiment/sbtm_n_$(n)_runs_$(num_runs)_start_$(start).jld2"
@assert load(path(ns[1], NUMRUNS, START), "num_runs") == NUMRUNS
@assert load(path(ns[1], NUMRUNS, START), "saveat") == SAVEAT
@assert load(path(ns[1], NUMRUNS, START), "start_time") == START

a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+START)/6)
true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * (2π*K)^(-3/2) * exp(-sum(abs2, x)/(2K))
true_marginal(x :: Number, K) = (a(K) + b(K)*x^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), [x])
true_marginal(x :: AbstractVector, K) = (a(K) + b(K)*x[1]^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), x)
true_slice(x, K) = true_pdf([x,0,0], K)
true_mean(K) = 0
true_cov(K) = I(3)

############ pdf plots ############

"plot the slice pdfs at time t"
function pdf_plot(solutions, labels, ε, time_index; num_runs=1, ci_alpha = 0.05)
    d = 3
    t = SAVEAT[time_index]
    n = num_particles(solutions[1][1]) ÷ num_runs
    plt = plot(title = "t = $(round(START+t, digits = 2)), n = $n", xlabel = "x", ylabel = "pdf(x)", ylim = (0, 0.05))
    pdf_range = range(-6, 6, length=500)
    label_ = "true"
    plot!(plt, pdf_range, x->true_slice(x, K(t)), label = label_)
    for (solution, label) in zip(solutions, labels)
        u = solution[time_index]
        lower, upper = kde_cis(pdf_range, x -> reconstruct_pdf(ε, [x,0,0], u), x->true_slice(x, K(t)), n*num_runs, ε, d; α = ci_alpha)
        kde = [reconstruct_pdf(ε, [x,0,0], u) for x in pdf_range]
        plot!(plt, pdf_range, kde, label = "$label, ε = $(round(ε, digits = 4)), $(1-ci_alpha) CI", ribbon = (kde .- lower, upper .- kde), fillalpha = 0.3)
    end
    plt
end

"Change n, plot slice pdfs at end time."
function slice_pdf_experiment(;time_index = 3)
    # ns = [200, 400, 1000, 2000, 4000, 10_000]
    ns = [15_000]
    plots = []
    for n in ns
        ε = rec_epsilon(n*NUMRUNS)
        solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
        solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
        pdf_plt = pdf_plot([solution_sbtm, solution_blob], ["sbtm", "blob"], ε, time_index, num_runs=NUMRUNS)
        push!(plots, pdf_plt)
    end
    plot(plots..., layout = (2,3), size = (1600, 900))
end

plt1 = slice_pdf_experiment(time_index = 1);
plt2 = slice_pdf_experiment(time_index = 2);
plt3 = slice_pdf_experiment(time_index = 3);
plt = plot(plt1, plt2, plt3, layout = (3,1), size = (1000, 1000))
savefig(plt, "plots/landau 3d slice pdfs combined, start $START")

############ Lp error plots ############

"Change n, plot Lp error of k-particles marginal at end time vs n."
function Lp_error_experiment(d=3; p=2, k=3, verbose = 0, kwargs...)
    # ns = [50, 100, 200, 400, 500, 1000, 2000, 4000, 10_000, 20_000]
    t_range = SAVEAT
    plots = []
    for (i,t) in enumerate(t_range)
        sbtm_errors = Float64[]
        blob_errors = Float64[]
        @views for n in ns
            # ε = rec_epsilon(n)
            solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            lp_error_sbtm = Lp_error_slice(solution_sbtm[i], x->true_pdf(x, K(t)); p=p, k=k, verbose = verbose, kwargs...)
            lp_error_blob = Lp_error_slice(solution_blob[i], x->true_pdf(x, K(t)); p=p, k=k, verbose = verbose, kwargs...)
            push!(sbtm_errors, lp_error_sbtm)
            push!(blob_errors, lp_error_blob)
        end
        Lp_errors_plot = Lp_error_plot(ns, [sbtm_errors, blob_errors], ["sbtm", "blob"], [:red, :green], t, d, "landau", k, p)
        push!(plots, Lp_errors_plot)
    end
    Lp_plots = plot(plots..., layout = (length(plots), 1), size = (1000, 1000))
end

L2_errror_plot = Lp_error_experiment(;k=3, verbose = 1, max_evals=5*10^4, xlim = 5, atol = 0.0005)
savefig(L2_errror_plot, "plots/landau 3d L2 error combined")

############ score plots ############

function score_experiment()
    ts = SAVEAT
    plt = plot(title = "landau NN error, $(NUMRUNS) runs, start $START, 1/n ∑ᵢ|s(Xᵢ) - ∇log ρ(Xᵢ)|^2", xlabel = "time", ylabel = "error", size = (1000, 600))
    for n in ns
        solution = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "solution")
        s_values = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "s_values")
        errors = zeros(length(ts))
        @views for (k,t) in enumerate(ts)
            xs = solution[k]
            ys = score(x -> true_pdf(x, K(t)), xs)
            errors[k] = sum(abs2, reshape(s_values[:,:,k,:], :, n*num_runs) .- ys)/n
        end
        plot!(plt, ts, errors, label = "n = $n", marker = :circle, markersize = 3, linewidth = 2)
        # plot(loss_plot, error_plot, layout = (1, 2), size = (1400, 600))
    end
    plt
end

score_plot = score_experiment()
savefig(score_plot, "plots/landau 3d NN error, start $START")

############ moments plots ############
empirical_first_moment(xs :: AbstractArray{T, 2}) where T = vec(mean(xs, dims = 2))
empirical_second_moment(xs :: AbstractArray{T, 2}) where T = xs * xs' / (num_particles(xs) - 1)
empirical_covariance(xs :: AbstractArray{T, 2}) where T = empirical_second_moment(xs .- empirical_first_moment(xs))
function average_covariance(xs :: AbstractArray{T, 2}, num_runs) where T 
    split_xs = split_into_runs(xs, num_runs)
    mean(empirical_covariance(run_i) for run_i in eachslice(split_xs, dims = 3))
end
split_into_runs(xs :: AbstractArray{T, 2}, num_runs) where T = reshape(xs, size(xs, 1), :, num_runs)

function moment_experiment(d=3)
    # ns = [50, 100, 200, 400, 500, 1000, 2000, 4000, 10_000, 20_000]
    t_range = SAVEAT
    plots = []
    for (i,t) in enumerate(t_range)
        means_sbtm = []
        means_blob = []
        covs_sbtm = []
        covs_blob = []
        @views for n in ns
            solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            push!(means_sbtm, mean(empirical_first_moment(solution_sbtm[i])))
            push!(means_blob, mean(empirical_first_moment(solution_blob[i])))
            push!(covs_sbtm, average_covariance(solution_sbtm[i], NUMRUNS))
            push!(covs_blob, average_covariance(solution_blob[i], NUMRUNS))
        end
        mean_plt = mean_diff_plot(ns, [means_sbtm, means_blob], true_mean(K(t)), ["sbtm", "blob"], [:red, :green], t, d, "landau")
        cov_norm_plot = covariance_diff_norm_plot(ns, [covs_sbtm, covs_blob], true_cov(K(t)), ["sbtm", "blob"], [:red, :green], t, d, "landau")
        cov_trace_plot = covariance_diff_trace_plot(ns, [covs_sbtm, covs_blob], true_cov(K(t)), ["sbtm", "blob"], [:red, :green], t, d, "landau")
        push!(plots, mean_plt, cov_norm_plot, cov_trace_plot)
    end
    plot(plots..., layout = (3, length(t_range)), size = (1800, 1000))
end

moment_plot = moment_experiment()
savefig(moment_plot, "plots/landau 3d moments")