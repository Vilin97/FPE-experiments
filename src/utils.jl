using Statistics, LinearAlgebra
using Distributions: MvNormal

# mollifier ϕ_ε
mol(ε, x) = exp(-norm(x)^2/ε)/sqrt((π*ε)^length(x)) # = pdf(MvNormal(ε/2*I(length(x))), x)
grad_mol(ε, x) = -2/ε*mol(ε, x) .* x
Mol(ε, x, xs) = sum( mol(ε, x - x_q) for x_q in eachslice(xs, dims=length(size(xs))) )

function moving_trap(N, num_samples, num_timestamps)
    d = 2 # dimension of each particle
    a = Float32(2.) # trap amplitude
    w = Float32(1.) # trap frequency
    α = Float32(.5) # repelling force
    Δts = Float32(0.01)*ones(Float32, num_timestamps) # time increments
      
    # define drift vector field b and diffusion matrix D
    # β(t) = a*[cos(π*w*0f0), sin(π*w*0f0)] # no-drift case
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

function attractive_origin(num_samples, num_timestamps)
    b(x,t) = -x
    D(x,t) = one(eltype(x))
    ρ(t) = MvNormal((1 - exp(-2*(t+0.1)))*I(2)) # -> MvNormal(I(2)) as t -> ∞
    ρ₀ = ρ(0)
    xs = convert(Array{Float32, 3}, reshape(rand(ρ₀, num_samples), 2, 1, num_samples))
    Δts = Float32(0.01)*ones(Float32, num_timestamps)

    xs, Δts, b, D, ρ₀, ρ
end