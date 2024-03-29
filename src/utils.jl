using Statistics, LinearAlgebra, Distributions, HCubature, Zygote, DifferentialEquations, CUDA, Flux, JLD2, LoopVectorization
import Distributions.gradlogpdf

############ fast linear algebra ############
"sum of squares of the elements of z"
function normsq(z)
    res = zero(eltype(z))
    for zi in z
        res += zi*zi
    end
    res
end

"sum of squares of the elements of z1 - z2"
function normsq(z1, z2)
    res = zero(eltype(z1))
    for i in eachindex(z1)
        res += (z1[i] - z2[i])^2
    end
    res
end

function fastdot(z, v)
    res = zero(eltype(z))
    for i in eachindex(z)
        res += z[i]*v[i]
    end
    res
end

############ analysis ############
### KDE ###
"1-α confidence interval"
function kde_ci(kde_approx_value, true_pdf_value, n, ε, d; α = 0.05, kernel_l2_squared = (4π)^(-d/2))
    h = (ε/2)^(1/2) # ε = 2 * h^2
    amplitude = sqrt(true_pdf_value * kernel_l2_squared / (n*h^d))
    pm = amplitude * quantile(Normal(0,1), 1-α/2)
    (kde_approx_value - pm, kde_approx_value + pm)
end

function kde_cis(kde_approx_values :: AbstractArray, true_pdf_values :: AbstractArray, n, ε, d; kwargs...)
    cis = [kde_ci(kde_approx_values[i], true_pdf_values[i], n, ε, d; kwargs...) for i in eachindex(kde_approx_values)]
    lower = [ci[1] for ci in cis]
    upper = [ci[2] for ci in cis]
    lower, upper
end

function kde_cis(xs, kde_fun :: Function, true_pdf :: Function, n, ε, d; kwargs...)
    kde_approx_values = kde_fun.(xs)
    true_pdf_values = true_pdf.(xs)
    kde_cis(kde_approx_values, true_pdf_values, n, ε, d; kwargs...)
end

### pdf reconstruction ###
"gaussian mollifier pdf(MvNormal(ε/2*I(length(x))), x)"
mol(ε, x) = exp(-sum(abs2, x)/ε)/sqrt((π*ε)^length(x)) # = pdf(MvNormal(ε/2*I(length(x))), x)
grad_mol(ε, x) = -2/ε*mol(ε, x) .* x
Mol(ε, x, xs :: AbstractArray{T, 3}) where T = sum( mol(ε, x .- x_q) for x_q in eachslice(xs, dims=3) )
function Mol(ε, x, xs::AbstractArray{T,2}) where T
    res = zero(eltype(xs))
    d = length(x)
    eps_recip = 1/ε
    for x_q in eachcol(xs)
        res += exp(-normsq(x,x_q)*eps_recip)
    end
    res/sqrt((π*ε)^d)
end
Mol(ε, x, xs :: AbstractVector) = sum( mol(ε, x - x_q) for x_q in xs )

reconstruct_pdf(ε, x, u :: AbstractVector) = Mol(ε, x, u)/length(u)
reconstruct_pdf(ε, x, u :: AbstractMatrix) = (@assert length(x) == get_d(u); Mol(ε, x, u)/size(u, 2))
reconstruct_pdf(ε, x, u :: AbstractArray{T, 3}) where T = reconstruct_pdf(ε, x, reshape(u, :, size(u, 3)))
reconstruct_pdf(x, u) = reconstruct_pdf(rec_epsilon(num_particles(u)), x, u)
marginal(dist :: MvNormal, k=1) = MvNormal(mean(dist)[1:k], cov(dist)[1:k, 1:k])

slice(f, d, k) = x -> f([x..., zeros(d-k)...])

### moments ###
empirical_first_moment(xs :: AbstractArray{T, 2}) where T = vec(mean(xs, dims = 2))
empirical_second_moment(xs :: AbstractArray{T, 2}) where T = xs * xs' / (num_particles(xs) - 1)
empirical_covariance(xs :: AbstractArray{T, 2}) where T = empirical_second_moment(xs .- empirical_first_moment(xs))
function average_covariance(xs :: AbstractArray{T, 2}, num_runs) where T 
    split_xs = split_into_runs(xs, num_runs)
    mean(empirical_covariance(run_i) for run_i in eachslice(split_xs, dims = 3))
end
split_into_runs(xs :: AbstractArray{T, 2}, num_runs) where T = reshape(xs, size(xs, 1), :, num_runs)

### entropies ###
# TODO: currently incorrect
"1/n ∑ᵢ [log rec_pdf(xᵢ) - log true_pdf(xᵢ)]"
function KL_divergence(u :: AbstractArray, true_pdf)
    n = num_particles(u)
    Z = sum(true_pdf, eachcol(u))
    sum(x -> log(Z/n/true_pdf(x)), eachcol(u))/n
end

function entropy(ε, u :: AbstractMatrix)
    n = num_particles(u)
    sum( log(reconstruct_pdf(ε, x, u)) for x in eachcol(u) )/n
end

### Lp error ###
function Lp_error_slice(u :: AbstractArray, true_pdf; k = 2, p = 2, verbose = 0, xlim = 10, rtol = 0.1, kwargs...)
    d = get_d(u)
    ε = rec_epsilon(num_particles(u))
    diff(x) = (reconstruct_pdf(ε, x, u) - true_pdf(x))^p 
    diff_slice = slice(diff, d, k)
    integrand, accuracy = hcubature(diff_slice, fill(-xlim, k), fill(xlim, k), rtol = rtol)
    if accuracy > rtol*abs(integrand)
        error("accuracy = $accuracy < $integrand*$rtol = Lp integrand*rtol")
    end
    verbose > 0 && (println("relative integration error ~ $(accuracy/abs(integrand))"))
    max(eps(), integrand)^(1/p)
end

function Lp_error(u :: AbstractArray, true_pdf; p = 2, verbose = 0, xlim = 10, max_evals = 2*10^4, rtol = 0.1, kwargs...)
    d = get_d(u)
    ε = rec_epsilon(num_particles(u))
    diff(x) = (reconstruct_pdf(ε, x, u) - true_pdf(x))^p 
    integrand, accuracy = hcubature(diff, fill(-xlim, d), fill(xlim, d), maxevals = max_evals, rtol = rtol)
    if accuracy > rtol*abs(integrand)
        error("accuracy = $accuracy < $integrand*$rtol = Lp integrand*rtol")
    end
    verbose > 0 && (println("relative integration error ~ $(accuracy/abs(integrand))"))
    max(eps(), integrand)^(1/p)
end

"Lp error between the reconstructed pdf and the true pdf at time t. pdf(x :: AbstractArray{T,k})"
function Lp_error_marginal(u_ :: AbstractArray, pdf; ε = 0.1, p = 2, verbose = 0, xlim = 10, k = 1, max_evals = 10^5, rtol = 0.05, kwargs...)
    n = num_particles(u_)
    d = get_d(u_)
    k = min(k, d)
    u = reshape(u_, d, n)[1:k, :]
    empirical_pdf(x) = reconstruct_pdf(ε, x, u)
    diff(x) = (abs(empirical_pdf(x) - pdf(x)))^p 
    error, accuracy = hcubature(diff, fill(-xlim, k), fill(xlim, k), maxevals = max_evals, rtol = rtol)
    verbose > 0 && (println("L$p error integration accuracy = $accuracy"))
    max(eps(), error)^(1/p)
end

function Lp_error_marginal(solution :: ODESolution, t_index, pdf; kwargs...)
    Lp_error_marginal(solution[t_index], pdf; kwargs...)
end

"true_solution(t) :: MvNormal"
function Lp_error_marginal(solution :: ODESolution, true_solution, t; k = 1, kwargs...)
    true_marginal = marginal(true_solution(t), k)
    Lp_error_marginal(solution(t), x -> pdf(true_marginal, x); k=k, kwargs...)
end

############ misc ############
num_particles(xs :: AbstractArray{T, 3}) where T = size(xs, 3)
num_particles(xs :: AbstractArray{T, 2}) where T = size(xs, 2)
num_particles(xs :: AbstractArray{T, 1}) where T = size(xs, 1)
get_d(xs :: AbstractArray{T, 3}) where T = size(xs, 1)*size(xs, 2)
get_d(xs :: AbstractArray{T, 2}) where T = size(xs, 1)
get_d(xs :: AbstractArray{T, 1}) where T = 1

"∇log ρ(x) for each column x of xs."
# score(ρ, xs) = mapslices(x -> gradlogpdf(ρ, x), xs, dims=1) # this is slow
score(ρ, xs :: CuArray) = gpu(score(ρ, Array(xs)))
score(ρ, xs :: AbstractArray{T,1}) where T = gradlogpdf(ρ, xs)
score(ρ, xs :: AbstractArray{T,2}) where T = reshape(hcat([gradlogpdf(ρ, @view xs[:,i]) for i in axes(xs, 2)]...), size(xs))
score(ρ, xs :: AbstractArray{T,3}) where T = reshape(hcat([gradlogpdf(ρ, @view xs[:,i,j]) for i in axes(xs, 2), j in axes(xs, 3)]...), size(xs))
gradlogpdf(f :: Function, x) = gradient(log ∘ f, x)[1]

split_into_runs(xs :: AbstractArray{T, 2}, num_runs) where T = reshape(xs, size(xs, 1), :, num_runs)

function log_fit(xs, ys)
    fit_log = Polynomials.fit(log.(xs), log.(ys), 1)
    slope = round(fit_log.coeffs[2], digits = 2)
    poly = exp.( fit_log.(log.(xs)) )
    slope, poly
end

############ FPE set-up ############
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

function pure_diffusion(d, n; dt = 1f-2, t_end = 2f0)
    b(x,t) = zero(x)
    D(x,t::T) where T = T(1.0)
    ρ(t) = MvNormal(2 * (t+1) * I(d))
    ρ₀ = ρ(0.)
    xs = Float32.(reshape(rand(ρ₀, n), d, n))
    tspan = (0f0, t_end)
    ts = tspan[1]:dt:tspan[2]

    xs, ts, b, D, ρ₀, ρ
end

function attractive_origin(d, n, dt = 0.005, t_end = 1.)
    b(x,t) = -x
    D(x,t::T) where T = T(1)
    ρ(t) = MvNormal((1 - exp(-2*(t+1)))*I(d))
    ρ₀ = ρ(0.)
    xs = reshape(rand(ρ₀, n), d, n)
    tspan = (0.0, t_end)
    ts = tspan[1]:dt:tspan[2]

    xs, ts, b, D, ρ₀, ρ
end

epsilon(d, n) = rec_epsilon(n, d)

######## Landau equation in 3D ########
"reconstruction epsilon = 2*h^2"
rec_epsilon(n, d = 3) = 2 * kde_bandwidth(n, d)^2
"kernel bandwidth h ~ n^(-1/(d+4))"
kde_bandwidth(n, d = 3) = n^(-1/(d+4)) / √2

function landau(d, n, start_time; dt = 1f-2, time_interval = 0.5f0)
    K(t) = 1 - exp(-(d-1)*(t+start_time)/12) # assumes C = 1 and B = 1/24
    f(x, t) = (2π*K(t))^(-d/2) * exp(-sum(abs2, x)/(2K(t))) * (((d+2)*K(t)-d)/(2K(t)) + (1-K(t))/(2K(t)^2)*norm(x)^2) # target density
    δ = 0.5 # how close the proposal distribution is to the target density
    M = 3 # upper bound on the ratio f/g in rejection sampling, good up to d=5
    xs = zeros(Float32, d, n)
    for i in 1:n
        xs[:, i] = rejection_sample(x -> f(x, 0), MvNormal(K(0.) * (1+δ) * I(d)), M)
    end
    tspan = (0.,time_interval)
    ts = tspan[1]:dt:tspan[2]

    xs, ts, f
end

function rejection_sample(target_density, proposal_dist, M)
    f = target_density
    g(x) = pdf(proposal_dist, x)
    while true
        x = rand(proposal_dist)
        if M * g(x) < f(x)
            error("M = $M is too low: $(M*g(x)) = Mg(x) > f(x) = $(f(x)) for x = $x.")
        end
        if rand() * M * g(x) < f(x) # accept with probability f(x)/Mg(x)
            return x
        end
    end
end

function landau_3d_f!(dxs, xs, score_values)
    dxs .= 0
    n = num_particles(xs)
    @turbo for p = 1:n
        Base.Cartesian.@nexprs 3 i -> dx_i = zero(eltype(dxs))
        for q = 1:n
            dotzv = zero(eltype(dxs))
            normsqz = zero(eltype(dxs))
            Base.Cartesian.@nexprs 3 i -> begin
                z_i = xs[i, p] - xs[i, q]
                v_i = score_values[i, q] - score_values[i, p]
                dotzv += z_i * v_i
                normsqz += z_i * z_i
            end
            Base.Cartesian.@nexprs 3 i -> begin
                dx_i += v_i * normsqz - dotzv * z_i
            end
        end
        Base.Cartesian.@nexprs 3 i -> begin
            dxs[i, p] += dx_i
        end
    end
    dxs ./= 24*n
    nothing
end