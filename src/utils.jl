using Statistics, LinearAlgebra
using Distributions
using HCubature

num_particles(xs :: AbstractArray{T, 3}) = size(xs, 3)
num_particles(xs :: AbstractArray{T, 2}) = size(xs, 2)
num_particles(xs :: AbstractArray{T, 1}) = size(xs, 1)

"∇log ρ(x) for each column x of xs."
# score(ρ :: MultivariateDistribution, xs) = mapslices(x -> gradlogpdf(ρ, x), xs, dims=1) # this is slow
score(ρ :: MultivariateDistribution, xs :: AbstractArray{T,1}) where T = gradlogpdf(ρ, xs)
score(ρ :: MultivariateDistribution, xs :: AbstractArray{T,2}) where T = reshape(hcat([gradlogpdf(ρ, @view xs[:,i]) for i in axes(xs, 2)]...), size(xs))
score(ρ :: MultivariateDistribution, xs :: AbstractArray{T,3}) where T = reshape(hcat([gradlogpdf(ρ, @view xs[:,i,j]) for i in axes(xs, 2), j in axes(xs, 3)]...), size(xs))

"gaussian mollifier pdf(MvNormal(ε/2*I(length(x))), x)"
mol(ε, x) = exp(-norm(x)^2/ε)/sqrt((π*ε)^length(x)) # = pdf(MvNormal(ε/2*I(length(x))), x)
grad_mol(ε, x) = -2/ε*mol(ε, x) .* x
Mol(ε, x, xs) = sum( mol(ε, x - x_q) for x_q in eachslice(xs, dims=length(size(xs))) )
Mol(ε, x, xs :: AbstractVector) = sum( mol(ε, x - x_q) for x_q in xs )

reconstruct_pdf(ε, x, u :: AbstractVector) = Mol(ε, x, u)/length(u)
reconstruct_pdf(ε, x, u :: AbstractMatrix) = Mol(ε, x, u)/size(u, 2)
reconstruct_pdf(ε, x, u :: AbstractArray{T, 3}) where T = reconstruct_pdf(ε, x, reshape(u, :, size(u, 3)))
marginal(dist :: MvNormal, k=1) = MvNormal(mean(dist)[1:k], cov(dist)[1:k, 1:k])

"Lp error between the reconstructed pdf and the true pdf at time t. If k < d, the marginals of the first k coordinates are compared."
function Lp_error(solution, true_solution, ε, t, d, n; p = 1, verbose = 0, xlim = 10, k = d)
    u = @view reshape(solution(t), d, n)[1:k, :]
    empirical_pdf(x) = reconstruct_pdf(ε, x, u)
    true_sol = marginal(true_solution(t), k)
    true_pdf(x) = pdf(true_sol, x)
    diff(x) = (empirical_pdf(x) - true_pdf(x))^p 
    error, accuracy = hcubature(diff, fill(-xlim, k), fill(xlim, k), maxevals = 10^5, atol = 0.001)
    verbose > 0 && (println("L$p error integration accuracy = $accuracy"))
    max(eps(), error)^(1/p)
end

function moving_trap(N, num_samples, num_timestamps)
    d = 2 # dimension of each particle
    a = Float32(2.) # trap amplitude
    w = Float32(1.) # trap frequency
    α = Float32(.5) # repelling force
    Δts = Float32(0.01)*ones(Float32, num_timestamps) # time increments
      
    # define drift vector field b and diffusion matrix D
    β(t) = a*[cos(π*w*t), sin(π*w*t)]
    function b(x, t)
        attract = β(t) .- x
        repel = α * (x .- mean(x, dims = 2))
        attract + repel
    end
    D(x, t :: T) where T = T(0.25)
    
    # draw samples
    ρ₀ = MvNormal(β(0.), 0.25f0*I(d))
    xs = convert(Array{Float32, 3}, reshape(rand(ρ₀, N*num_samples), d, N, num_samples))

    # positions of moving trap
    target = hcat(β.(vcat(0., cumsum(Δts)))...)

    xs, Δts, b, D, ρ₀, target, a, w, α, β
end

function pure_diffusion(d, n, dt = 0.005, t_end = 2.)
    b(x,t) = zero(x)
    D(x,t::T) where T = T(1.0)
    ρ(t) = MvNormal(2. * (t+1.) * I(d))
    ρ₀ = ρ(0.)
    xs = reshape(rand(ρ₀, n), d, 1, n)
    tspan = (0.0, t_end)
    ts = tspan[1]:dt:tspan[2]

    xs, ts, b, D, ρ₀, ρ
end

function attractive_origin(d, n, dt = 0.005, t_end = 1.)
    b(x,t) = -x
    D(x,t::T) where T = T(1.0)
    ρ(t) = MvNormal((1 - exp(-2*(t+1.)))*I(d))
    ρ₀ = ρ(0.)
    xs = reshape(rand(ρ₀, n), d, 1, n)
    tspan = (0.0, t_end)
    ts = tspan[1]:dt:tspan[2]

    xs, ts, b, D, ρ₀, ρ
end

"set ε ~ n^(k/d) to account for particles getting sparser with dimension. At c = 1, k = 1, epsilon(2,4000) = 0.05"
epsilon(d, n, c = 1., k=1) = c * 4000. ^(k/2) / (20. * n^(k/d))



######## Landau equation in 3D ########
epsilon_landau(n, L = 4.) = 0.64 * (2L/n)^1.98 # h = 2L/n, and ε = 0.64 h^1.98

function landau_3D(n, dt = 0.005, t_end = 0.5)
    t_start = 5.5
    K(t) = 1 - exp(-(t+t_start)/6)
    f(x, K) = pdf(MvNormal(K * I(3)), x) * ((5K-3)/(2K) + (1-K)/(2K^2)*norm(x)^2) # target density
    δ = 0.3 # how close the proposal distribution is to the target density
    M = 2 # upper bound on the ratio f/g in rejection sampling
    xs = zeros(3, n)
    for i in 1:n
        xs[:, i] = rejection_sample(x -> f(x, K(0.)), MvNormal(K(0.)/(1-δ) * I(3)), M)
    end
    tspan = (t_start, t_end)
    ts = tspan[1]:dt:tspan[2]

    xs, ts, f, K
end

function rejection_sample(target_density, proposal_dist, M)
    f = target_density
    g(x) = pdf(proposal_dist, x)
    while true
        x = rand(proposal_dist)
        if rand() * M * g(x) < f(x) # accept with probability f(x)/Mg(x)
            return x
        end
    end
end