using Distributions, Plots, TimerOutputs, Polynomials, JLD2, DifferentialEquations, OrdinaryDiffEq, Flux, Roots
using Random: seed!

plotly()
include("utils.jl")
include("blob.jl")
include("sbtm.jl")

"Solve the pure diffusion problem of dimension d with n particles."
function solve_diffusion(d, n)
    seed!(1234)
    xs, ts, b, D, ρ₀, ρ = pure_diffusion(d, n)
    ε = 0.053
    @timeit "blob" solution_blob = blob(xs, ts, b, D; ε = ε)
    @timeit "sbtm" solution_sbtm = sbtm(xs, ts, b, D; ρ₀ = ρ₀, optimiser = Adam(10^-2))

    solution_blob, solution_sbtm, ρ, ts, ε
end

"plot the marginal pdfs at time t"
function pdf_plot(solutions, labels, true_solution, t, ε; true_regularized = nothing)
    d, = size(true_solution(t))
    u = reshape(solutions[1](t), d, :)
    n = size(u, 2)
    plt = plot(title = "$(d)d marginal, t = $(round(t, digits = 2)), n = $n", xlabel = "x", ylabel = "pdf(x)", ylim = (0, 0.5))
    pdf_range = range(-6, 6, length=100)
    label_ = (d==1 && n==200) ? "true" : nothing
    true_marginal = marginal(true_solution(t))
    plot!(plt, pdf_range, [pdf(true_marginal, [x]) for x in pdf_range], label = label_)
    if true_regularized !== nothing
        label_ = (d==1 && n==200) ? "true regularized" : nothing
        plot!(plt, pdf_range, [pdf(marginal(true_regularized(t)), [x]) for x in pdf_range], label = label_)
    end
    for (solution, label) in zip(solutions, labels)
        u = reshape(solution(t), d, n)[1, :]
        label_ = (d==1 && n==200) ? label : nothing
        plot!(plt, pdf_range, [reconstruct_pdf(ε, x, u) for x in pdf_range], label = label_)
    end
    plt
end

analytic_entropies(ρ, ts) = entropy.(ρ.(ts))
function empirical_entropy(ε, u :: AbstractVector)
    n = length(u)
    -sum( log(Mol(ε, x, u)/n) for x in u )/n
end
empirical_entropies(ε, u) = empirical_entropy.(ε, u)
function entropy_plot(solutions, labels, true_solution, ts)
    plt = plot(title = "entropy comparison", xlabel = "t", ylabel = "entropy", size = (1000, 300))
    anal_ent = analytic_entropies(true_solution, ts)
    for (solution, label) in zip(solutions, labels)
        emp_ent = empirical_entropies(ε, reshape.(solution(ts).u, :))
        plot!(plt, ts, emp_ent, label = label)
    end
    plot!(plt, ts, anal_ent, label = "true entropy")
end

"Keep n fixed, plot pdfs, entropy and L2 error at evenly spaced times."
function fixed_n_experiment(n = 1000)
    solution_blob, solution_sbtm, ρ, ts, ε = solve_diffusion(1, n)
    plots = []
    for t in range(ts[1], ts[end], length=12)
        push!(plots, pdf_plot([solution_blob, solution_sbtm], ["blob, eps=$ε", "sbtm"], ρ, t))
    end
    println("plotting pdfs")
    pdf_plot = plot(plots..., size = (1400, 900))
    println("plotting entropy")
    ent_plot = entropy_plot([solution_blob, solution_sbtm], ["blob, eps=$ε", "sbtm"], ρ, ts)
    println("plotting L2 error")
    l2_plot = l2_error_plot([solution_blob, solution_sbtm], ["blob, eps=$ε", "sbtm"], ρ, range(ts[1], ts[end], length = 10))
    big_plot = plot(entplot, l2_plot, pdf_plot, layout = (3, 1), size = (1800, 1000))
    pdf_plot, ent_plot, l2_plot, big_plot
end

"Change n, plot Lp error of k-particles marginal at end time vs n."
function Lp_error_experiment(d; p=2, k=1, verbose = 0, experiment = pure_diffusion, experiment_name = "pure_diffusion")
    @show d
    ε = 0.053
    ns = [50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 4000]
    _, ts, _, _, _, ρ = experiment(d, ns[1])
    t_range = range(0.5, ts[end], length = 2)
    plots = []
    for (i,t) in enumerate(t_range)
        blob_errors = Float64[]
        sbtm_errors = Float64[]
        for n in ns
            solution_blob = JLD2.load("$(experiment_name)_experiment/blob_d_$(d)_n_$(n).jld2", "solution")
            solution_sbtm = JLD2.load("$(experiment_name)_experiment/sbtm_d_$(d)_n_$(n).jld2", "solution")
            push!(blob_errors, Lp_error(solution_blob, ρ, ε, t, d, n; verbose = verbose, p = p, k=k))
            push!(sbtm_errors, Lp_error(solution_sbtm, ρ, ε, t, d, n; verbose = verbose, p = p, k=k))
        end
        Lp_errors_plot = Lp_error_plot(ns, [blob_errors, sbtm_errors], ["blob", "sbtm"], [:blue, :red], t, d, experiment_name, k, p)
        push!(plots, Lp_errors_plot)
    end
    Lp_plots = plot(plots..., layout = (1, length(plots)))
end

"Change n, plot Lp error of k-particles marginal at end time vs n."
function marginal_pdf_experiment(ds; experiment = pure_diffusion, experiment_name = "pure_diffusion")
    pdf_plots_ = []
    for d in ds
        @show d
        ε = 0.053
        ns = [200, 1000, 2000, 4000]
        _, ts, _, _, _, ρ = experiment(d, ns[1])
        plots = []
        for n in ns
            t_plot = ts[end]
            f(σ,t) = σ + epsilon(d,n)log(σ) - (2t + tr(cov(ρ(0)))/d + epsilon(d,n)log(tr(cov(ρ(0)))/d))
            σ(t) = solve(ZeroProblem(σ -> f(σ,t), tr(cov(ρ(t)))/d )) # the variance of the regularized solution
            true_regularized(t) = MvNormal(σ(t) * I(d))
            solution_sbtm = JLD2.load("$(experiment_name)_experiment/sbtm_d_$(d)_n_$(n).jld2", "solution")
            solution_blob1 = JLD2.load("$(experiment_name)_experiment/blob_d_$(d)_n_$(n)_eps_$(epsilon(d,n)).jld2", "solution")
            solution_blob2 = JLD2.load("$(experiment_name)_experiment/blob_simple_d_$(d)_n_$(n)_eps_$(epsilon(d,n)).jld2", "solution")
            pdf_plt = pdf_plot([solution_sbtm, solution_blob1, solution_blob2], ["sbtm", "blob", "blob simple"], ρ, t_plot, ε; true_regularized = true_regularized)
            push!(plots, pdf_plt)
        end
        plt_ = plot(plots..., layout = (1, length(plots)))
        push!(pdf_plots_, plt_)
    end
    plot(pdf_plots_..., layout = (length(pdf_plots_), 1), size = (1400, 800))
end

# Lp_error_plots = []
# for d in [1,2,5,10]
#     error_plt = Lp_error_experiment(d, experiment = attractive_origin, experiment_name = "attractive_origin")
#     push!(Lp_error_plots, error_plt)
# end
# plt=plot(Lp_error_plots..., layout = (length(Lp_error_plots), 1), size = (1400, 800))

plt_ = marginal_pdf_experiment([1,2,3,5,10], experiment = pure_diffusion, experiment_name = "pure_diffusion")
