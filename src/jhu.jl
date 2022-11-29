using Distributions, Zygote, LinearAlgebra
using Distributions: pdf, MvNormal

# mollifier ϕ_ε
mol(ε, x) = exp(-sum(x.^2)/ε)/sqrt((π*ε)^length(x))
Mol(ε, x, xs) = sum( mol(ε, x - x_q) for x_q in eachslice(xs, dims=3) )

function propagate(ε :: Number, x, xs, t, Δt, b, D)
    d1 = gradient(x -> Mol(ε, x, xs), x)[1]/Mol(ε, x, xs)
    d2 = sum( gradient(x -> mol(ε, x - x_q), x)[1]/Mol(ε, x_q, xs) for x_q in eachslice(xs, dims=3) )
    x + Δt * (b(x, t) - D(x,t) * (d1 + d2))
end

function jhu(xs, Δts, b, D, ε)
    trajectories = zeros(eltype(xs), size(xs)..., 1+length(Δts)) # trajectories[:, i, j, k] is particle i of sample j at time k
    trajectories[:, :, :, 1] = xs
    jhu!(trajectories, Δts, b, D, ε)
    trajectories
end

function jhu!(trajectories, Δts, b, D, ε)
    t = zero(eltype(Δts))
    for (k, Δt) in enumerate(Δts)
        xs_k = trajectories[:, :, :, k]
        for (j, x) in enumerate(eachslice(xs_k, dims = 3))
            trajectories[:, :, j, k+1] = propagate(ε, x, xs, t, Δt, b, D)
        end
        t += Δt
    end
end

function moving_trap()
    d = 2 # dimension of each particle
    N = 3 # number of particles in a sample
    a = Float32(2.) # trap amplitude
    w = Float32(1.) # trap frequency
    α = Float32(0.5) # repelling force
    num_samples = 2 # number of samples
    Δts = 0.01*ones(Float32, 200) # time increments
      
    # define drift vector field b and diffusion matrix D
    β(t) = a*Float32[cos(π*w*t), sin(π*w*t)]
    function b(x, t)
        attract = β(t) .- x
        repel = α * (x .- mean(x, dims = 2))
        attract + repel
    end
    D(x, t) = Float32(0.25)
    
    # draw samples
    ρ₀ = MvNormal(β(0.), 0.001*I(d))
    xs = convert(Array{Float32, 3}, reshape(rand(ρ₀, N*num_samples), d, N, num_samples))
    xs, Δts, b, D, β
end

xs, Δts, b, D, β = moving_trap()
trajectories = jhu(xs, Δts, b, D, 1.0)
target = hcat(β.(vcat(0., cumsum(Δts)))...)
animate_2d(trajectories; target = target)