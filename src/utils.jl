using Statistics, LinearAlgebra, Distributions, HCubature, Zygote
import Distributions.gradlogpdf

"sum of squares of the elements of z"
function normsq(z)
    res = zero(eltype(z))
    @fastmath for zi in z
        res += zi*zi
    end
    res
end

function fastdot(z, v)
    res = zero(eltype(z))
    @fastmath for i in eachindex(z)
        res += z[i]*v[i]
    end
    res
end

num_particles(xs :: AbstractArray{T, 3}) where T = size(xs, 3)
num_particles(xs :: AbstractArray{T, 2}) where T = size(xs, 2)
num_particles(xs :: AbstractArray{T, 1}) where T = size(xs, 1)

"∇log ρ(x) for each column x of xs."
# score(ρ, xs) = mapslices(x -> gradlogpdf(ρ, x), xs, dims=1) # this is slow
score(ρ, xs :: AbstractArray{T,1}) where T = gradlogpdf(ρ, xs)
score(ρ, xs :: AbstractArray{T,2}) where T = reshape(hcat([gradlogpdf(ρ, @view xs[:,i]) for i in axes(xs, 2)]...), size(xs))
score(ρ, xs :: AbstractArray{T,3}) where T = reshape(hcat([gradlogpdf(ρ, @view xs[:,i,j]) for i in axes(xs, 2), j in axes(xs, 3)]...), size(xs))
gradlogpdf(f :: Function, x) = gradient(log ∘ f, x)[1]

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
function Lp_error(solution, ε, t, d, n; p = 1, verbose = 0, xlim = 10, k = 1, marginal_pdf)
    u = @view reshape(solution(t), d, n)[1:k, :]
    empirical_pdf(x) = reconstruct_pdf(ε, x, u)
    diff(x) = (empirical_pdf(x) - marginal_pdf(x))^p 
    error, accuracy = hcubature(diff, fill(-xlim, k), fill(xlim, k), maxevals = 10^5, atol = 0.001)
    verbose > 0 && (println("L$p error integration accuracy = $accuracy"))
    max(eps(), error)^(1/p)
end

function Lp_error(solution, true_solution, ε, t, d, n; p = 1, verbose = 0, xlim = 10, k = 1)
    true_marginal = marginal(true_solution(t), k)
    Lp_error(solution, ε, t, d, n; p = p, verbose = verbose, xlim = xlim, k = k, marginal_pdf = x -> pdf(true_marginal, x))
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

function landau(n; dt = 0.01)
    K(t) = 1 - exp(-(t+5.5)/6)
    # f(x, t) = (k=1 - exp(-(t+5.5)/6); (2π*k)^(-3/2) * exp(-norm(x)^2/(2k)) * ((5k-3)/(2k) + (1-k)/(2k^2)*norm(x)^2)) # target density
    f(x, t) = (k=1 - exp(-(t+20.)/6); (2π*k)^(-3/2) * exp(-norm(x)^2/(2k)) * ((5k-3)/(2k) + (1-k)/(2k^2)*norm(x)^2)) # target density
    δ = 0.3 # how close the proposal distribution is to the target density
    M = 2 # upper bound on the ratio f/g in rejection sampling
    xs = zeros(3, n)
    for i in 1:n
        xs[:, i] = rejection_sample(x -> f(x, K(0.)), MvNormal(K(0.)/(1-δ) * I(3)), M)
    end
    tspan = (0.,0.5)
    ts = tspan[1]:dt:tspan[2]

    xs, ts, f
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