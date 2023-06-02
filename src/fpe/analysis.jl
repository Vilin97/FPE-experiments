using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux
using Random: seed!

include("blob.jl")
include("sbtm.jl")
include("../utils.jl")
include("../plotting_utils.jl")

const START = 1
const EXPNAME = "diffusion"
const SAVEAT = [0., 1., 2.]
const NUMRUNS = 10
const ds = [2,3,5,10]
const ns = [2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
const labels = labels
const colors = [:red, :green, :purple]
path(d, n, label) = "diffusion_experiment/$(label)_d_$(d)_n_$(n)_runs_$(NUMRUNS).jld2"
solutions(d, n) = [JLD2.load(path(d, n, label), "solution") for label in labels]
for n in ns, d in ds, label in labels
    @assert load(path(d, n, label), "num_runs") == NUMRUNS
    @assert load(path(d, n, label), "saveat") == SAVEAT
end

true_dist(t, d) = MvNormal(2*(t+START)*I(d))
true_marginal(x, t; k = 1) = (@assert k == length(x); (4π * (t+START))^(-k/2) * exp(-sum(abs2, x)/(4 * (t+START))))
true_pdf(x, t) = (d = length(x); (4π * (t+START))^(-d/2) * exp(-sum(abs2, x)/(4 * (t+START))))
true_slice(x, t, d) = true_pdf(x_slice(x,d), t)
true_mean(t, d) = zeros(d)
true_cov(t, d) = 2*(t+START)*I(d)
x_slice(x :: Number, d; y = zeros(d-1)) = [x, y...]

############ Scatter plots ############
function scatter_plot(solutions, labels; time_index, max_particles = 5000)
    markersize = 4
    d = get_d(solutions[1][1])
    t = SAVEAT[time_index]
    n = num_particles(solutions[1][1]) ÷ NUMRUNS
    n_plot = min(n, max_particles)
    initial_sample = solutions[1][1][:, 1:n_plot]
    plt = plot(title = "d = $d, n = $n, t = $(round(START+t, digits = 2)), n_plot = $n_plot", xlabel = "x", ylabel = "y", size = (1300, 1300))
    for (solution, label) in zip(solutions, labels)
        u = solution[time_index][1:2, 1:n_plot]
        scatter!(plt, u[1,:], u[2,:], label = "$label", markersize = markersize)
    end
    scatter!(plt, initial_sample[1,:], initial_sample[2,:], label = "initial sample", markersize = markersize)
    plt
end

function scatter_experiment(ds_ = ds)
    plots = Vector{Plots.Plot}(undef, length(ds_))
    n = 2500
    time_index = 3
    for (j, d) in enumerate(ds_)
        plots[j] = scatter_plot(solutions(d, n), labels; time_index = time_index)
    end
    plt = plot(plots..., layout = (2,2), size = (1500, 1500))
end

scatter_plt = scatter_experiment();
savefig(scatter_plt, "plots/$(EXPNAME) scatter, start $START")

############ pdf plots ############

function slice_pdf_plot(solutions, labels; time_index)
    d = get_d(solutions[1][1])
    t = SAVEAT[time_index]
    n = num_particles(solutions[1][1]) ÷ NUMRUNS
    ε = 4*rec_epsilon(n*NUMRUNS, d)
    plt = plot(title = "d = $d, n = $n, t = $(round(START+t, digits = 2)), ε = $(round(ε, digits = 4))", xlabel = "x", ylabel = "slice pdf(x)")
    pdf_range = range(-6, 6, length=200)
    plot!(plt, pdf_range, x->true_slice(x, t, d), label = "true")
    for (solution, label) in zip(solutions, labels)
        u = solution[time_index]
        kde = [reconstruct_pdf(ε, x_slice(x,d), u) for x in pdf_range]
        plot!(plt, pdf_range, kde, label = "$label")
    end
    plt
end

function marginal_pdf_plot(solutions, labels; time_index)
    d = get_d(solutions[1][1])
    t = SAVEAT[time_index]
    n = num_particles(solutions[1][1]) ÷ NUMRUNS
    ε = 2*rec_epsilon(n*NUMRUNS, d)
    plt = plot(title = "d = $d, n = $n, t = $(round(START+t, digits = 2)), ε = $(round(ε, digits = 4))", xlabel = "x", ylabel = "marginal pdf(x)", ylim = (0,true_marginal(0, 0)))
    pdf_range = range(-6, 6, length=200)
    plot!(plt, pdf_range, x->true_marginal(x, t), label = "true")
    for (solution, label) in zip(solutions, labels)
        u = solution[time_index]
        kde = [reconstruct_pdf(ε, x, u[1,:]) for x in pdf_range]
        plot!(plt, pdf_range, kde, label = "$label")
    end
    plt
end

function pdf_experiment(ns_ = ns[end], time_indices = [1,2,3], ds_ = ds; slice = true)
    plots = Array{Plots.Plot, 3}(undef, length(ns_), length(time_indices), length(ds_))
    for (i,n) in enumerate(ns_), (j,time_index) in enumerate(time_indices), (k, d) in enumerate(ds_)
        @show n, time_index, d
        if slice
            plots[i,j,k] = slice_pdf_plot(solutions(d, n), labels; time_index = time_index)
        else
            plots[i,j,k] = marginal_pdf_plot(solutions(d, n), labels; time_index = time_index)
        end
    end
    plt = plot(plots..., layout = (length(ds_), length(time_indices)*length(ns_)), size = (1800, 1000))
end

plt = pdf_experiment(slice = false);
savefig(plt, "plots/$(EXPNAME) marginal pdfs combined, start $START")

############ entropy plots ############
function entropy_plot(solutions, labels)
    ts = SAVEAT
    d = get_d(solutions[1][1])
    n = num_particles(solutions[1][1]) ÷ NUMRUNS
    ε = epsilon(d, n)
    plt = plot(title = "entropy comparison, d = $d, n = $n", xlabel = "t", ylabel = "entropy", size = (1000, 300))
    plot!(plt, ts, -Distributions.entropy.(true_dist.(ts, d)), label = "true entropy") 
    for (solution, label) in zip(solutions, labels)
        entropies = similar(ts)
        for tidx in eachindex(ts)
            u = split_into_runs(solution[tidx], NUMRUNS)
            entropies[tidx] = sum(entropy(ε, @view u[:,:,run]) for run in 1:NUMRUNS)/NUMRUNS
        end
        plot!(plt, ts, entropies, label = label)
    end
    plt
end

function entropy_experiment(ns_ = ns[1:3], ds_ = ds[1:end-1])
    plots = Matrix{Plots.Plot}(undef, length(ns_), length(ds_))
    for (i,n) in enumerate(ns_), (j,d) in enumerate(ds_)
        @show n, d
        plots[i,j] = entropy_plot(solutions(d, n), labels)
    end
    plt = plot(plots..., layout = (length(ds_), length(ns_)), size = (1800, 900))
end

entropy_plt = entropy_experiment()
savefig(entropy_plt, "plots/$(EXPNAME) entropy, start $START")

############ Lp error plots ############

"Change n, plot Lp error of k-particles marginal at end time vs n."
function Lp_error_experiment(ds_ = ds, ns_ = ns; p=2, k=2, verbose = 0, kwargs...)
    ts = SAVEAT
    plots = Matrix{Plots.Plot}(undef, length(ts), length(ds_))
    for (i,t) in enumerate(ts), (j,d) in enumerate(ds_)
        errors = [Vector{eltype(ts)}(undef, length(ns_)) for _ in labels]
        @views for (n_idx, n) in enumerate(ns_), (l, label) in enumerate(labels)
            @show t, d, n, label
            solution = JLD2.load(path(d, n, label), "solution")[i]
            split_sol = reshape(solution, d, n, NUMRUNS)
            lp_error = sum(Lp_error_marginal(split_sol[:,:,run], x->true_marginal(x, t; k=k); k=k, p=p, verbose = verbose, kwargs...) for run in 1:NUMRUNS) / NUMRUNS
            errors[l][n_idx] = lp_error
        end
        Lp_errors_plot = Lp_error_plot(ns_, errors, labels, colors, t, d, "$(EXPNAME)", k, p)
        plots[i, j] = Lp_errors_plot
    end
    Lp_plots = plot(plots..., layout = (length(ds_), length(ts)), size = (1800, 1000))
end

@time L2_error_plot = Lp_error_experiment(ds, ns; verbose = 0, xlim = 2, rtol = 2 * 0.02, k = 2);
savefig(L2_error_plot, "plots/$(EXPNAME) L2 error combined")

############ KL divergence plots ############
function KL_divergence_experiment(;verbose = 0)
    t_range = SAVEAT
    plots = []
    for (i,t) in enumerate(t_range)
        sbtm_errors = Float64[]
        blob_errors = Float64[]
        @views for n in ns
            verbose > 0 && println("t = $t, n = $n")
            solution_sbtm = JLD2.load("$(EXPNAME)_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            solution_blob = JLD2.load("$(EXPNAME)_experiment/blob_n_$(n)_runs_$(NUMRUNS)_start_$(START).jld2", "solution")
            kl_div_sbtm = KL_divergence(solution_sbtm[i], x->true_pdf(x, K(t)))
            kl_div_blob = KL_divergence(solution_blob[i], x->true_pdf(x, K(t)))
            push!(sbtm_errors, kl_div_sbtm)
            push!(blob_errors, kl_div_blob)
        end
        plts = kl_div_plot(ns,labels, [:red, :green], t, "$(EXPNAME)")
        push!(plots, plts)
    end
    plot(plots..., layout = (length(plots), 1), size = (1000, 1000))
end

# kl_plot = KL_divergence_experiment(verbose=1)
# savefig(kl_plot, "plots/$(EXPNAME) 3d KL divergence")

############ score plots ############

function score_error_experiment()
    ts = SAVEAT
    plt = plot(title = "$(EXPNAME) NN R^2, $(NUMRUNS) runs, start $START", xlabel = "time", ylabel = "R^2 score", size = (1000, 600))
    for n in ns
        solution = JLD2.load("$(EXPNAME)_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "solution")
        combined_models = JLD2.load("$(EXPNAME)_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "models")
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
# score_error_plot = score_error_experiment()
# savefig(score_plot, "plots/$(EXPNAME) 3d NN R^2, start $START")

function score_slice_experiment(ns_indices = [length(ns)-1, length(ns)], time_indices = [1,2,3])
    plots = Matrix{Plots.Plot}(undef, length(ns_indices), length(time_indices))
    for (i,n_index) in enumerate(ns_indices)
        n = ns[n_index]
        combined_models = JLD2.load("$(EXPNAME)_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "models")
        solution = JLD2.load("$(EXPNAME)_experiment/sbtm_n_$(n)_runs_$(NUMRUNS)_start_$START.jld2", "solution")
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
# score_slice_plot = score_slice_experiment()
# savefig(score_slice_plot, "plots/$(EXPNAME) 3d score slice, start $START")

############ moments plots ############
empirical_first_moment(xs :: AbstractArray{T, 2}) where T = vec(mean(xs, dims = 2))
empirical_second_moment(xs :: AbstractArray{T, 2}) where T = xs * xs' / (num_particles(xs) - 1)
empirical_covariance(xs :: AbstractArray{T, 2}) where T = empirical_second_moment(xs .- empirical_first_moment(xs))
function average_covariance(xs :: AbstractArray{T, 2}, num_runs) where T 
    split_xs = split_into_runs(xs, num_runs)
    mean(empirical_covariance(run_i) for run_i in eachslice(split_xs, dims = 3))
end
split_into_runs(xs :: AbstractArray{T, 2}, num_runs) where T = reshape(xs, size(xs, 1), :, num_runs)

function moment_experiment(ds_=ds, ns_ = ns)
    ts = SAVEAT
    mean_plots = Matrix{Plots.Plot}(undef, length(ts), length(ds_))
    cov_norm_plots = Matrix{Plots.Plot}(undef, length(ts), length(ds_))
    cov_trace_plots = Matrix{Plots.Plot}(undef, length(ts), length(ds_))
    for (i,t) in enumerate(ts), (j,d) in enumerate(ds_)
        means = [Vector{Vector{eltype(ts)}}(undef, length(ns_)) for _ in labels]
        covs = [Vector{Matrix{eltype(ts)}}(undef, length(ns_)) for _ in labels]
        @views for (n_idx, n) in enumerate(ns_), (l, label) in enumerate(labels)
            solution = JLD2.load(path(d, n, label), "solution")
            means[l][n_idx] = empirical_first_moment(solution[i])
            covs[l][n_idx] = average_covariance(solution[i], NUMRUNS)
        end
        mean_plt = mean_diff_plot(ns_, means, true_mean(t,d), labels, colors, t, d, "$(EXPNAME)")
        cov_norm_plot = covariance_diff_norm_plot(ns_, covs, true_cov(t,d), labels, colors, t, d, "$(EXPNAME)")
        cov_trace_plot = covariance_diff_trace_plot(ns_, covs, true_cov(t,d), labels, colors, t, d, "$(EXPNAME)")
        mean_plots[i,j] = mean_plt
        cov_norm_plots[i,j] = cov_norm_plot
        cov_trace_plots[i,j] = cov_trace_plot
    end
    mean_plot = plot(mean_plots..., layout = (length(ds_), length(ts)), size = (1800, 1000))
    cov_norm_plot = plot(cov_norm_plots..., layout = (length(ds_), length(ts)), size = (1800, 1000))
    cov_trace_plot = plot(cov_trace_plots..., layout = (length(ds_), length(ts)), size = (1800, 1000))
    return mean_plot, cov_norm_plot, cov_trace_plot
end

mean_plt, cov_norm_plt, cov_trace_plt = moment_experiment()
savefig(mean_plt, "plots/$(EXPNAME) mean")
savefig(cov_norm_plt, "plots/$(EXPNAME) cov norm")
savefig(cov_trace_plt, "plots/$(EXPNAME) cov trace")