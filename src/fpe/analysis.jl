using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux

include("blob.jl")
include("sbtm.jl")
include("../utils.jl")
include("../plotting_utils.jl")
include("../analysis_utils.jl")

ENV["GKSwstype"] = "nul" # no gui

const SAVEPLOTS = true
const VERBOSE = 2
const START = 1
const EXPNAME = "diffusion"
const SAVEAT = [0., 1., 2.]
const NUMRUNS = 20
const ds = [2,3,5,10]
const ns = [2_500, 5_000, 10_000, 20_000, 40_000, 80_000]
const labels = ["exact", "sbtm", "blob"]
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
scatter_plt = scatter_experiment();
SAVEPLOTS && savefig(scatter_plt, "plots/$(EXPNAME) scatter, start $START")

############ pdf plots ############
pdf_plt = pdf_experiment(slice = false);
SAVEPLOTS && savefig(pdf_plt, "plots/$(EXPNAME) marginal pdfs combined, start $START")

############ Lp error plots ############
@time L2_error_plot = Lp_error_experiment(ds, ns; verbose = 0, xlim = 2, rtol = 2 * 0.02, k = 2);
SAVEPLOTS && savefig(L2_error_plot, "plots/$(EXPNAME) L2 error combined")

############ moments plots ############
mean_plt, cov_norm_plt, cov_trace_plt = moment_experiment();
SAVEPLOTS && savefig(mean_plt, "plots/$(EXPNAME) mean")
SAVEPLOTS && savefig(cov_norm_plt, "plots/$(EXPNAME) cov norm")
SAVEPLOTS && savefig(cov_trace_plt, "plots/$(EXPNAME) cov trace")

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
        VERBOSE > 1 && @show n, d
        plots[i,j] = entropy_plot(solutions(d, n), labels)
    end
    plt = plot(plots..., layout = (length(ds_), length(ns_)), size = (1800, 900))
end

# entropy_plt = entropy_experiment()
# SAVEPLOTS && savefig(entropy_plt, "plots/$(EXPNAME) entropy, start $START")

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
# SAVEPLOTS && savefig(kl_plot, "plots/$(EXPNAME) 3d KL divergence")

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
# SAVEPLOTS && savefig(score_plot, "plots/$(EXPNAME) 3d NN R^2, start $START")

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
# SAVEPLOTS && savefig(score_slice_plot, "plots/$(EXPNAME) 3d score slice, start $START")