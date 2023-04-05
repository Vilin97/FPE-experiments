# rejection sampling test

using Plots
include("../src/utils.jl")

n = 100000
t_start = 5.5
a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+t_start)/6)
# f(x, K) = pdf(MvNormal(K * I(3)), x) * (a(K) + b(K)*norm(x)^2)
analytic_marginal(x, K) = (a(K) + b(K)*x^2 + (1-K)/K) * pdf(MvNormal(K*I(1)), [x])
xs, ts, _ = landau(n)
x1s = xs[1, :]
plt=histogram(x1s, normed=true, label="sampled");
plot!(plt, x -> analytic_marginal(x,K(0.)), label="analytic");
plot!(plt, title = "Rejection sampling for Landau", ylabel = "marginal density", xlabel = "x")
