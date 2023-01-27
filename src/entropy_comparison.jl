using Distributions, Plots
include("utils.jl")
plotly()

# compare analytic entropy to empirical entropy of a sample
function empirical_entropy(ε, sample :: AbstractVector)
    n = length(sample)
    -sum( log(Mol(ε, x, sample)/n) for x in sample )/n
end

sigmas = 0.1:0.1:5.
epss = 2.0 .^ (-7:2:3)
plt = plot(xlabel = "σ = variance of Normal", ylabel = "entropy", title = "entropy vs σ", size = (1000, 500), legend = :bottomright)
for eps in epss
    @show eps
    local ents = Float64[]
    for σ in sigmas
        local z = Normal(0., sqrt(σ))
        ent = mean([empirical_entropy(eps, sample) for sample in eachcol(rand(z, 100, 100))])
        push!(ents, ent)
    end
    plot!(plt, sigmas, ents, label = "empirical ε = $eps")
end

plot!(plt, sigmas, entropy.(Normal.(0., sqrt.(sigmas))), label = "analytic", lw = 4, marker = :diamond)