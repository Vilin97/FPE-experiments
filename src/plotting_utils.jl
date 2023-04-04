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

function plot_losses(losses)
    epochs = size(losses, 1)
    p = plot(vec(losses), title = "Score approximation", xaxis = "epochs", yaxis = "loss", label = "training loss")
    scatter!(p, 1:epochs:length(vec(losses)), vec(losses)[1:epochs:end], label = "discrete time propagation", marker = true)
end

function Lp_error_plot(ns, errors, labels, colors, t, d, experiment_name, k, p)
    Lp_errors_plot = plot(title = "$(d)d $experiment_name $k-marginal L$p error at t = $t, log-log", ylabel = "L$p error from true pdf", xaxis = :log, yaxis = :log, xlabel = "number of particles")
    dense_ns = range(ns[1], ns[end], length = 1000)
    for (error, label, color) in zip(errors, labels, colors)
        error_log = log.(error)
        fit_log = Polynomials.fit(log.(ns), error_log, 1)
        slope = round(fit_log.coeffs[2], digits = 2)
        plot!(Lp_errors_plot, ns, error, label = (d==1 && t==0.5) ? "t = $t, $label" : nothing, marker = :circle, color = color)
        poly = exp.( fit_log.(log.(dense_ns)) )
        plot!(Lp_errors_plot, dense_ns, poly, label = "d = $d, t = $t, $label slope $slope", color = color, opacity = 0.4)
    end
    Lp_errors_plot
end