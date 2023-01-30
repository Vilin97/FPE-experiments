using Test, Distributions, Plots
using Random: seed!
plotly()
include("../src/jhu.jl")
include("../src/sbtm.jl")

seed!(1234)
b(x,t) = zero(x)
D(x,t) = 1.0
ρ(t) = Normal(0., sqrt(2. * (t+1.)))
ρ₀ = ρ(0.)
n = 40^2
xs = rand(ρ₀, 1, 1, n)
dt = 0.005
tspan = (0.0, 0.5)
ts = tspan[1]:dt:tspan[2]
num_ts = Int(tspan[2]/dt)
Δts = repeat([dt], num_ts)

ε = 0.053
println("Solving jhu with ε = $ε")
@timed (_, solution_jhu), t = jhu(xs, Δts, b, D; ε = ε)
println("Took $t seconds")
println("Solving sbtm")
@timed (_, extras), t = sbtm(xs, Δts, b, D; ρ₀ = MvNormal(2. * I(1)), optimiser = Adam(10^-2))
println("Took $t seconds")
solution_sbtm = extras["solution"]

reconstruct_pdf(ε, x, u) = Mol(ε, x, u)/length(u)

function pdf_plots(solutions, labels, true_solution, t)
    plt = plot(title = "t = $(round(t, digits = 2)), #particles = $n", xlabel = "x", ylabel = "pdf(x)", ylim = (0, 0.5))
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
    print("Plotting entropy")
    plt = plot(title = "entropy comparison", xlabel = "t", ylabel = "entropy", size = (1000, 300))
    anal_ent = analytic_entropies(true_solution, ts)
    true_sample_entropy = mean(empirical_entropies(ε, [rand(true_solution(t), n) for t in ts]) for _ in 1:100)
    for (solution, label) in zip(solutions, labels)
        emp_ent = empirical_entropies(ε, reshape.(solution(ts).u, :))
        plot!(plt, ts, emp_ent, label = label)
    end
    plot!(plt, ts, anal_ent, label = "true entropy")
    plot!(plt, ts, true_sample_entropy, label = "true sample entropy")
end

plots = []
for t in range(tspan[1], tspan[2], length=12)
    push!(plots, pdf_plots([solution_jhu, solution_sbtm], ["jhu, eps=$ε", "sbtm"], ρ, t))
end
# entplot = entropy_plot([solution_jhu, solution_sbtm], ["jhu, eps=$ε", "sbtm"], ρ, ts)
pdf_plot = plot(plots..., size = (1500, 1000))
# big_plot = plot(entplot, pdf_plot, layout = (2, 1), size = (1800, 1000))
