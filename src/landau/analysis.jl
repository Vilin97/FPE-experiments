using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux

include("blob.jl")
include("sbtm.jl")
include("../utils.jl")
include("../plotting_utils.jl")
include("../analysis_utils.jl")

ENV["GKSwstype"] = "nul" # no gui

const SAVEPLOTS = true
const VERBOSE = 2
const START = 6
const EXPNAME = "landau"
const SAVEAT = [0f0, 0.25f0, 0.5f0]
const NUMRUNS = 10
const ds = [3]
# const ns = [2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
const ns = [2_500, 5_000, 10_000, 20_000, 40_000]
const labels = ["exact", "sbtm", "blob"]
const colors = [:red, :green, :purple]
path(d, n, label) = "$(EXPNAME)_experiment/$(label)_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2"
solutions(d, n) = [JLD2.load(path(d, n, label), "solution") for label in labels]
for n in ns, d in ds, label in labels
    @assert load(path(d, n, label), "num_runs") == NUMRUNS
    @assert load(path(d, n, label), "saveat") ≈ SAVEAT
end

a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+START)/6)
true_pdf(x, t) = (a(K(t)) + b(K(t))*sum(abs2, x)) * (2π*K(t))^(-3/2) * exp(-sum(abs2, x)/(2K(t)))
true_mean(t, d) = zeros(d)
true_cov(t, d) = I(d)
# TODO: extend to k = 2,3
true_marginal(x, t; k = 1) = (@assert 1 == length(x); (4π * (t+START))^(-1/2) * exp(-sum(abs2, x)/(4 * (t+START)))) * (a(K(t)) + b(K(t)) * sum(abs2, x) + 2b(K(t)))
# TODO: the plots look weird. Something might be wrong
############ Scatter plots ############
scatter_plt = scatter_experiment(ds);
SAVEPLOTS && savefig(scatter_plt, "plots/$(EXPNAME) scatter, start $START")

############ pdf plots ############
pdf_plt = pdf_experiment(ns[end], [1,2,3], ds, slice = false);
SAVEPLOTS && savefig(pdf_plt, "plots/$(EXPNAME) marginal pdfs combined, start $START")

############ Lp error plots ############
@time L2_error_plot = Lp_error_experiment(ds, ns; verbose = 0, xlim = 2, rtol = 2 * 0.02, k = 1);
SAVEPLOTS && savefig(L2_error_plot, "plots/$(EXPNAME) L2 error combined")

############ moments plots ############
mean_plt, cov_norm_plt, cov_trace_plt = moment_experiment(ds, ns);
SAVEPLOTS && savefig(mean_plt, "plots/$(EXPNAME) mean")
SAVEPLOTS && savefig(cov_norm_plt, "plots/$(EXPNAME) cov norm")
SAVEPLOTS && savefig(cov_trace_plt, "plots/$(EXPNAME) cov trace")











############ KL divergence plots ############
function KL_divergence_experiment(;verbose = 0)
    t_range = SAVEAT
    plots = []
    for (i,t) in enumerate(t_range)
        sbtm_errors = Float64[]
        blob_errors = Float64[]
        @views for n in ns
            verbose > 0 && println("t = $t, n = $n")
            solution_sbtm = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            solution_sbtm = [solution_sbtm[1], solution_sbtm[3], solution_sbtm[4]] # remove t = 0.01 solution
            solution_blob = JLD2.load("landau_experiment/blob_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            kl_div_sbtm = KL_divergence(solution_sbtm[i], x->true_pdf(x, K(t)))
            kl_div_blob = KL_divergence(solution_blob[i], x->true_pdf(x, K(t)))
            push!(sbtm_errors, kl_div_sbtm)
            push!(blob_errors, kl_div_blob)
        end
        plts = kl_div_plot(ns, [sbtm_errors, blob_errors], ["sbtm", "blob"], [:red, :green], t, "landau")
        push!(plots, plts)
    end
    plot(plots..., layout = (length(plots), 1), size = (1000, 1000))
end

kl_plot = KL_divergence_experiment(verbose=1)
savefig(kl_plot, "plots/landau 3d KL divergence")

############ score plots ############

function score_error_experiment()
    ts = SAVEAT
    plt = plot(title = "landau NN R^2, $(NUMRUNS) runs, start $START", xlabel = "time", ylabel = "R^2 score", size = (1000, 600))
    for n in ns
        solution = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "solution")
        solution = [solution[1], solution[3], solution[4]] # remove t = 0.01 solution
        combined_models = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "models")
        combined_models = combined_models[[1,3,4], :] # remove t = 0.01 model
        errors = zeros(length(ts))
        s_values = zeros(Float32, get_d(solution[1]), n, NUMRUNS)
        @views for (k,t) in enumerate(ts)
            @show n, t
            xs = solution[k]
            split_xs = reshape(xs, size(xs, 1), :, NUMRUNS)
            for run in 1:NUMRUNS
                @views s_values[:,:,run] .= combined_models[k, run](split_xs[:,:,run])
            end
            ys = score(x -> true_pdf(x, K(t)), xs)
            errors[k] = sum(abs2, reshape(s_values, :, n*NUMRUNS) .- ys)/sum(abs2, ys .- mean(ys, dims = 2))
        end
        plot!(plt, ts, 1 .- errors, label = "n = $n", marker = :circle, markersize = 3, linewidth = 2)
    end
    plt
end
score_error_plot = score_error_experiment()
savefig(score_plot, "plots/landau 3d NN R^2, start $START")

function score_slice_experiment(ns_indices = [length(ns)-1, length(ns)], time_indices = [1,2,3])
    plots = Matrix{Plots.Plot}(undef, length(ns_indices), length(time_indices))
    for (i,n_index) in enumerate(ns_indices)
        n = ns[n_index]
        combined_models = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "models")
        combined_models = combined_models[[1,3,4], :] # remove t = 0.01 model
        solution = JLD2.load("landau_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "solution")
        solution = solution[[1,3,4]] # remove t = 0.01 solution
        for (j, time_index) in enumerate(time_indices)
            @show n, time_index
            t = SAVEAT[time_index]
            plt = plot(title = "score slice at t = $(t), n = $n", xlabel = "x", ylabel = "score");
            plot!(plt, -4:0.01:4, x -> score(y -> true_pdf(y, K(t)), [x,0.,0.])[1], label="true");
            scatter!(plt, solution[time_index][1,1:n ÷ 5], randn(n ÷ 5)./10, label="particles", markersize = 1, color = :black)
            for run in 1:NUMRUNS
                s = combined_models[time_index, run]
                plot!(plt, -4:0.01:4, x -> s(Float32[x,0.,0.])[1], label="run $run");
            end
            plots[i,j] = plt
        end
    end
    plt = plot(plots..., layout = (length(time_indices), length(ns_indices)), size = (1800, 1000))
end
score_slice_plot = score_slice_experiment()
savefig(score_slice_plot, "plots/landau 3d score slice, start $START")

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