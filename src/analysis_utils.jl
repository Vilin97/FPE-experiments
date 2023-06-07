# Assumes existence of a bunch of constants and functions

# const SAVEPLOTS = true
# const VERBOSE = 2
# const START = 1
# const EXPNAME = "diffusion"
# const SAVEAT = [0., 1., 2.]
# const NUMRUNS = 20
# const ds = [2,3,5,10]
# const ns = [2_500, 5_000, 10_000, 20_000, 40_000, 80_000]
# const labels = ["exact", "sbtm", "blob"]
# const colors = [:red, :green, :purple]
# path(d, n, label) = "diffusion_experiment/$(label)_d_$(d)_n_$(n)_runs_$(NUMRUNS).jld2"
# solutions(d, n) = [JLD2.load(path(d, n, label), "solution") for label in labels]
# for n in ns, d in ds, label in labels
#     @assert load(path(d, n, label), "num_runs") == NUMRUNS
#     @assert load(path(d, n, label), "saveat") == SAVEAT
# end

# true_marginal(x, t; k = 1) = (@assert k == length(x); (4π * (t+START))^(-k/2) * exp(-sum(abs2, x)/(4 * (t+START))))
# true_pdf(x, t) = (d = length(x); (4π * (t+START))^(-d/2) * exp(-sum(abs2, x)/(4 * (t+START))))
# true_mean(t, d) = zeros(d)
# true_cov(t, d) = 2*(t+START)*I(d)

############ helpers ############
true_slice(x, t, d) = true_pdf(x_slice(x,d), t)
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
    VERBOSE > 0 && println("scatter experiment")
    plots = Vector{Plots.Plot}(undef, length(ds_))
    n = 2500
    time_index = 3
    for (j, d) in enumerate(ds_)
        plots[j] = scatter_plot(solutions(d, n), labels; time_index = time_index)
    end
    plt = plot(plots..., layout = (2,2), size = (1500, 1500))
end

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
    VERBOSE > 0 && println("pdf experiment")
    plots = Array{Plots.Plot, 3}(undef, length(ns_), length(time_indices), length(ds_))
    for (i,n) in enumerate(ns_), (j,time_index) in enumerate(time_indices), (k, d) in enumerate(ds_)
        VERBOSE > 1 && @show n, time_index, d
        if slice
            plots[i,j,k] = slice_pdf_plot(solutions(d, n), labels; time_index = time_index)
        else
            plots[i,j,k] = marginal_pdf_plot(solutions(d, n), labels; time_index = time_index)
        end
    end
    plt = plot(plots..., layout = (length(ds_), length(time_indices)*length(ns_)), size = (1800, 1000))
end

############ Lp error plots ############

"Change n, plot Lp error of k-particles marginal at end time vs n."
function Lp_error_experiment(ds_ = ds, ns_ = ns; p=2, k=2, verbose = 0, kwargs...)
    VERBOSE > 0 && println("Lp error experiment")
    ts = SAVEAT
    plots = Matrix{Plots.Plot}(undef, length(ts), length(ds_))
    for (i,t) in enumerate(ts), (j,d) in enumerate(ds_)
        errors = [Vector{eltype(ts)}(undef, length(ns_)) for _ in labels]
        @views for (n_idx, n) in enumerate(ns_), (l, label) in enumerate(labels)
            VERBOSE > 1 && @show t, d, n, label
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

############ moments plots ############
function moment_experiment(ds_=ds, ns_ = ns)
    VERBOSE > 0 && println("moment experiment")
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