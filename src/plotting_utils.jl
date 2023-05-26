using Plots

function animate_2d(trajectories, title, Δts; samples = 1, fps = 5, target = zeros(2,length(Δts)+1), plot_every = 1)
    xmax = maximum([maximum(target[1,:]), maximum(abs.(trajectories[1,:,samples,:]))]) + 0.5
    ymax = maximum([maximum(target[2,:]), maximum(abs.(trajectories[2,:,samples,:]))]) + 0.5
    @show xmax, ymax
    ts = round.(vcat(zero(Δts[1]), cumsum(Δts)), digits = 4)
    plot_size = (700,700)
    p = scatter(trajectories[1,:,samples,1], trajectories[2,:,samples,1], 
                title = "$title, time $(ts[1])",
                label = nothing, 
                color=RGB(0.0, 0.0, 1.0), 
                xlims = (-xmax, xmax), ylims = (-ymax, ymax),
                markersize = 3, size = plot_size);
    target !== nothing && scatter!(p, [target[1, 1]], [target[2, 1]], markershape = :star, label = "target", color = :red)
    
    anim = @animate for k ∈ axes(trajectories, 4)[1:plot_every:end]
        λ = k/size(trajectories, 4)
        red = λ > 0.5 ? 2. *(λ - 0.5) : 0.
        green = 1. - abs(1. - 2. * λ)
        blue = λ < 0.5 ? 2. * (0.5-λ) : 0.
        target !== nothing && scatter!(p, [target[1, k]], [target[2, k]], markershape = :star, color = :red, label = nothing)
        scatter!(p, vec(trajectories[1,:,samples,k]), vec(trajectories[2,:,samples,k]), 
                  title = "$title, time $(ts[k])",
                  label = nothing, 
                  color = RGB(red, green, blue), 
                  xlims = (-xmax, xmax), ylims = (-ymax, ymax),
                  markersize = 3, size = plot_size)
  end
  gif(anim, "$(title)_anim_fps$fps.gif", fps = fps)
end

function plot_losses(losses :: AbstractMatrix; kwargs...)
    p = plot()
    plot_losses!(p, losses; kwargs...)
end

function plot_losses!(p, losses :: AbstractMatrix; label = nothing, kwargs...)
    epochs = size(losses, 1)
    plot!(p, vec(losses), title = "Score approximation training loss", xaxis = "epochs", yaxis = "loss", label = label, kwargs...)
    scatter!(p, 1:epochs:length(vec(losses)), vec(losses)[1:epochs:end], marker = true, label = nothing)
end

"ns = #particles, errors = kl divergences, t = time"
function kl_div_plot(ns, errors, labels, colors, t, experiment_name)
    plt = plot(title = "$experiment_name KL divergence, t = $t, log-log", ylabel = "KL divergence from true pdf", xaxis = :log, yaxis = :log, xlabel = "number of particles")
    for (error, label, color) in zip(errors, labels, colors)
        error_log = log.(error)
        fit_log = Polynomials.fit(log.(ns), error_log, 1)
        slope = round(fit_log.coeffs[2], digits = 2)
        plot!(plt, ns, error, label = "t = $t, $label", marker = :circle, color = color)
        poly = exp.( fit_log.(log.(ns)) )
        plot!(plt, ns, poly, label = "$label slope $slope", color = color, opacity = 0.4)
    end
    plt
end

"ns = #particles, errors = Lp errors, t = time, d = dimension, k = #dimensions used, p = Lp norm"
function Lp_error_plot(ns, errors, labels, colors, t, d, experiment_name, k, p)
    Lp_errors_plot = plot(title = "$(d)d $experiment_name L$p error, k = $k, t = $t, log-log", ylabel = "L$p error from true pdf", xaxis = :log, yaxis = :log, xlabel = "number of particles")
    for (error, label, color) in zip(errors, labels, colors)
        error_log = log.(error)
        fit_log = Polynomials.fit(log.(ns), error_log, 1)
        slope = round(fit_log.coeffs[2], digits = 2)
        plot!(Lp_errors_plot, ns, error, label = (d==1 && t==0.5) ? "t = $t, $label" : nothing, marker = :circle, color = color)
        poly = exp.( fit_log.(log.(ns)) )
        plot!(Lp_errors_plot, ns, poly, label = "$label slope $slope", color = color, opacity = 0.4)
    end
    Lp_errors_plot
end

"ns = #particles, t = time, d = dimension"
function mean_diff_plot(ns, means, true_mean, labels, colors, t, d, experiment_name)
    plt = plot(title = "$(d)d $experiment_name mean, t = $t", ylabel = "mean position 2-norm", xlabel = "number of particles", xscale = :log, yscale = :log)
    true_means = [true_mean for _ in 1:length(ns)]
    @assert size(true_mean, 1) == d
    @assert size(means[1][1], 1) == d
    for (mean, label, color) in zip(means, labels, colors)
        ys = norm.(mean .- true_means)
        fit_log = Polynomials.fit(log.(ns), log.(ys), 1)
        slope = round(fit_log.coeffs[2], digits = 2)
        poly = exp.( fit_log.(log.(ns)) )
        plot!(plt, ns, poly, label = nothing, color = color, opacity = 0.4)
        plot!(plt, ns, norm.(mean .- true_means), label = "$label, slope $slope", marker = :circle, color = color)
    end
    plt
end

"ns = #particles, t = time, d = dimension"
function covariance_diff_norm_plot(ns, covs, true_cov, labels, colors, t, d, experiment_name)
    plt = plot(title = "$(d)d $experiment_name covariance difference, 2-norm, t = $t", ylabel = "norm of diff from true cov", xlabel = "number of particles", xscale = :log, yscale = :log)
    true_covs = [true_cov for _ in 1:length(ns)]
    @assert size(true_cov, 1) == d
    @assert size(true_cov, 2) == d
    @assert size(covs[1][1], 2) == d
    for (covariance, label, color) in zip(covs, labels, colors)
        ys = norm.(covariance .- true_covs)
        fit_log = Polynomials.fit(log.(ns), log.(ys), 1)
        slope = round(fit_log.coeffs[2], digits = 2)
        poly = exp.( fit_log.(log.(ns)) )
        plot!(plt, ns, poly, label = nothing, color = color, opacity = 0.4)
        plot!(plt, ns, norm.(covariance .- true_covs), label = "$label, slope $slope", marker = :circle, color = color)
    end
    plt
end

"ns = #particles, t = time, d = dimension"
function covariance_diff_trace_plot(ns, covs, true_cov, labels, colors, t, d, experiment_name)
    plt = plot(title = "$(d)d $experiment_name covariance difference, |trace|, t = $t", ylabel = "|trace| of diff from true cov", xlabel = "number of particles", xscale = :log, yscale = :log)
    true_covs = [true_cov for _ in 1:length(ns)]
    for (covariance, label, color) in zip(covs, labels, colors)
        ys = abs.(tr.(covariance .- true_covs))
        fit_log = Polynomials.fit(log.(ns), log.(ys), 1)
        slope = round(fit_log.coeffs[2], digits = 2)
        poly = exp.( fit_log.(log.(ns)) )
        plot!(plt, ns, poly, label = nothing, color = color, opacity = 0.4)
        plot!(plt, ns, abs.(tr.(covariance .- true_covs)), label = "$label, slope $slope", marker = :circle, color = color)
    end
    plt
end