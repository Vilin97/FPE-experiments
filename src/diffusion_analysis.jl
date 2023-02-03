using Distributions, Plots, TimerOutputs, Polynomials
using Random: seed!
plotly()
include("../src/utils.jl")
include("../src/jhu.jl")
include("../src/sbtm.jl")

function solve_diffusion_1d(n)
    seed!(1234)
    xs, ts, Δts, b, D, ρ₀, ρ = diffusion_1d(n) 
    ε = 0.053
    @timeit "jhu" _, solution_jhu = jhu(xs, Δts, b, D; ε = ε)
    @timeit "sbtm" _, extras = sbtm(xs, Δts, b, D; ρ₀ = MvNormal(2. * I(1)), optimiser = Adam(10^-2))
    solution_sbtm = extras["solution"]

    solution_jhu, solution_sbtm, ρ, ts, ε
end

reconstruct_pdf(ε, x, u) = Mol(ε, x, u)/length(u)

function pdf_plots(solutions, labels, true_solution, t)
    plt = plot(title = "t = $(round(t, digits = 2)), #particles = $(length(solutions[1][end]))", xlabel = "x", ylabel = "pdf(x)", ylim = (0, 0.5))
    pdf_range = range(-6, 6, length=100)
    for (solution, label) in zip(solutions, labels)
        u = reshape(solution(t), :)
        label_ = t == 0.0 ? label : nothing
        plot!(plt, pdf_range, [reconstruct_pdf(ε, x, u) for x in pdf_range], label = label_)
    end
    label_ = t == 0.0 ? "true" : nothing
    plot!(plt, pdf_range, pdf.(true_solution(t), pdf_range), label = label_)
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

function l2_error_plot(solutions, labels, true_solution, ts)
    plt = plot(title = "L2 error comparison, n = $(length(solutions[1][end]))", xlabel = "t", ylabel = "error", size = (1000, 300))
    pdf_range = range(-6, 6, length=100)
    for (solution, label) in zip(solutions, labels)
        l2_errors = []
        for t in ts
            u = reshape(solution(t), :)
            pdf_diff = [reconstruct_pdf(ε, x, u) for x in pdf_range] .- [pdf(true_solution(t), x) for x in pdf_range]
            l2_error = norm(pdf_diff) * sqrt(step(pdf_range))
            push!(l2_errors, l2_error)
        end
        plot!(plt, ts, l2_errors, label = label, marker = :circle)
    end
    plt
end

function fixed_n_experiment(n = 1000)
    solution_jhu, solution_sbtm, ρ, ts, ε = solve_diffusion_1d(n)
    plots = []
    for t in range(ts[1], ts[end], length=12)
        push!(plots, pdf_plots([solution_jhu, solution_sbtm], ["jhu, eps=$ε", "sbtm"], ρ, t))
    end
    println("plotting pdfs")
    pdf_plot = plot(plots..., size = (1400, 900))
    println("plotting entropy")
    ent_plot = entropy_plot([solution_jhu, solution_sbtm], ["jhu, eps=$ε", "sbtm"], ρ, ts)
    println("plotting l2 error")
    l2_plot = l2_error_plot([solution_jhu, solution_sbtm], ["jhu, eps=$ε", "sbtm"], ρ, range(ts[1], ts[end], length = 10))
    big_plot = plot(entplot, l2_plot, pdf_plot, layout = (3, 1), size = (1800, 1000))
    pdf_plot, ent_plot, l2_plot, big_plot
end

function L2_error_experiemnt()
    jhu_errors = Float64[]
    sbtm_errors = Float64[]
    ns = [50, 100, 200, 500, 1000, 2000, 4000, 6000, 8000]
    reset_timer!()
    for n in ns
        @show n
        @timeit "n = $n" solution_jhu, solution_sbtm, ρ, ts, ε = solve_diffusion_1d(n)
        push!(jhu_errors, L2_error(solution_jhu, ρ, ε, ts[end], 1, n))
        push!(sbtm_errors, L2_error(solution_sbtm, ρ, ε, ts[end], 1, n))
    end
    print_timer()
    jhu_errors_log = log.(jhu_errors)
    sbtm_errors_log = log.(sbtm_errors)
    jhu_fit_log = Polynomials.fit(log.(ns), jhu_errors_log, 1)
    sbtm_fit_log = Polynomials.fit(log.(ns), sbtm_errors_log, 1)
    jhu_slope = round(jhu_fit_log.coeffs[2], digits = 2)
    sbtm_slope = round(sbtm_fit_log.coeffs[2], digits = 2)
    l2_plot = plot(log.(ns), [jhu_errors_log sbtm_errors_log], label = ["jhu" "sbtm"], title = "1d diffusion L2 error, log-log", xlabel = "n = $ns", ylabel = "L2 error from true pdf", size = (1000, 600), marker = :circle)
    dense_ns = range(ns[1], ns[end], length = 1000)
    plot!(l2_plot, log.(dense_ns), [jhu_fit_log.(log.(dense_ns)) sbtm_fit_log.(log.(dense_ns))], label = ["jhu fit, slope $jhu_slope" "sbtm fit, slope $sbtm_slope"])
    jhu_errors, sbtm_errors, l2_plot
end