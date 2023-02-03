using Statistics, LinearAlgebra
using Distributions: MvNormal

# mollifier ϕ_ε
mol(ε, x) = exp(-norm(x)^2/ε)/sqrt((π*ε)^length(x)) # = pdf(MvNormal(ε/2*I(length(x))), x)
grad_mol(ε, x) = -2/ε*mol(ε, x) .* x
Mol(ε, x, xs) = sum( mol(ε, x - x_q) for x_q in eachslice(xs, dims=length(size(xs))) )
Mol(ε, x, xs :: AbstractVector) = sum( mol(ε, x - x_q) for x_q in xs )

reconstruct_pdf(ε, x, u :: AbstractMatrix) = Mol(ε, x, u)/size(u, 2)
reconstruct_pdf(ε, x, u :: AbstractArray{T, 3}) where T = reconstruct_pdf(ε, x, reshape(u, :, size(u, 3)))

"L2 error between the reconstructed pdf and the true pdf at time t"
function L2_error(solution, true_solution, ε, t, d, n; xlim = 3, h = 0.1)
    pdf_range = collect.(Iterators.product(fill(-xlim:h:xlim, d)...))
    u = reshape(solution(t), d, n)
    pdf_diff = [reconstruct_pdf(ε, x, u) for x in pdf_range] .- [pdf(true_solution(t), x) for x in pdf_range]
    l2_error = norm(pdf_diff) * sqrt(h^d)
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

function attractive_origin(num_samples, num_timestamps; Δt = 0.01)
    b(x,t) = -x
    D(x,t) = one(eltype(x))
    t₀ = 1.
    ρ(t) = MvNormal((1 - exp(-2*(t+t₀)))*I(2)) # -> MvNormal(I(2)) as t -> ∞
    ρ₀ = ρ(0)
    xs = convert(Array{Float32, 3}, reshape(rand(ρ₀, num_samples), 2, 1, num_samples))
    Δts = Float32(Δt)*ones(Float32, num_timestamps)

    xs, Δts, b, D, ρ₀, ρ
end

function diffusion_1d(n, dt = 0.005)
    b(x,t) = zero(x)
    D(x,t) = 1.0
    ρ(t) = MvNormal(2. * (t+1.) * I(1))
    ρ₀ = ρ(0.)
    xs = reshape(rand(ρ₀, n), 1, 1, n)
    dt = 0.005
    tspan = (0.0, 0.5)
    ts = tspan[1]:dt:tspan[2]
    num_ts = Int(tspan[2]/dt)
    Δts = repeat([dt], num_ts)

    xs, ts, Δts, b, D, ρ₀, ρ
end