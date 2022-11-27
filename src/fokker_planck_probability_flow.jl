# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, Plots
using Distributions: MvNormal, logpdf
using Zygote: gradient
using Flux.Optimise: Adam
using Flux: params, train!
using Statistics: mean

maybewrap(x) = x
maybewrap(x::T) where {T <: Number} = T[x]

# divergence of vector field s at x
divergence(s, x) = sum(gradient(x -> s(x)[i], x)[1][i] for i in 1:length(x))
loss(s, xs) = mean(sum(s(x).^2) + 2.0*divergence(s, x) for x in xs)
loss(s, xs :: Array{T, 3}) where T =  loss(s, eachslice(xs, dims = 3)) 

score(ρ, x) = sum(logpdf(ρ, x))
propagate(x, t, Δt, b, D, s) = x + Δt * b(x, t) + D(x, t)*s(x)

function custom_train!(s, xs; optimiser = Adam(5. * 10^-3), num_steps = 25)
    θ = params(s)
    for _ in 1:num_steps
        grads = gradient(() -> loss(s, xs), θ)
        Flux.update!(optimiser, θ, grads)
    end
end

function initialize_s!(s, ρ₀, xs; optimiser = Adam(10^-4), ε = 10^-4)
    θ = params(s)
    ys = gradient(x -> score(ρ₀, x), xs)[1]
    ys_sum_squares = sum(ys.^2)
    square_error(s) = sum( (s(xs) - ys).^2 / ys_sum_squares )
    while square_error(s) > ε
        grads = gradient(() -> square_error(s), θ)
        Flux.update!(optimiser, θ, grads)
    end
end

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ → Rᵈ
D   : Rᵈ → Rᵈˣᵈ
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm(xs, Δts, b, D, s)
    trajectories = zeros(size(xs)..., 1+length(Δts)) # trajectories[:, i, j, k] is particle i of sample j at time k
    trajectories[:, :, :, 1] = xs
    sbtm!(trajectories, Δts, b, D, s)
    trajectories
end

function sbtm!(trajectories, Δts, b, D, s)
    t = zero(eltype(Δts))
    for (k, Δt) in enumerate(Δts)
        @show k
        xs_k = trajectories[:, :, :, k]
        custom_train!(s, xs_k)
        for (j, x) in enumerate(eachslice(xs_k, dims = 3))
            trajectories[:, :, j, k+1] = propagate(x, t, Δt, b, D, s)
        end
        t += Δt
    end
end

# learning to animate
function animate_2d(trajectories; samples = 1, fps = 5)
    p = scatter(trajectories[1,:,samples,1], trajectories[2,:,samples,1], 
                title = "time 0",
                label = nothing, 
                color=RGB(0.0, 0.0, 1.0), 
                xlims = (-1.5, 1.5), ylims = (-1.5, 1.5),
                markersize = 3);
    anim = @animate for k ∈ axes(trajectories, 4)
        λ = k/size(trajectories, 4)
        red = λ > 0.5 ? 2. *(λ - 0.5) : 0.
        green = 1. - abs(1. - 2. * λ)
        blue = λ < 0.5 ? 2. * (0.5-λ) : 0.
        scatter!(p, vec(trajectories[1,:,samples,k]), vec(trajectories[2,:,samples,k]), 
                  title = "time $k",
                  label = nothing, 
                  color = RGB(red, green, blue), 
                  xlims = (-1.5, 1.5), ylims = (-1.5, 1.5),
                  markersize = 3)
  end
  gif(anim, "anim_fps$fps.gif", fps = fps)
end

# experimenting with all particles attracting to the origin
function attractive_origin()
    d = 2
    N = 1
    num_samples = 9
    Δts = 0.01*ones(50)
    b(x, t) = -x
    D(x, t) = 0.1
    s = Chain(
      Dense(d => 50, relu),
      Dense(50 => d))
    xs = reshape(Float64[-1  0  1 -1  0  1 -1  0  1;
                         -1 -1 -1  0  0  0  1  1  1], d, N, num_samples);
    print("Initializing s...")
    initialize_s!(s, MvNormal(zeros(d), I(d)), xs, ε = 10^-2)
    xs, Δts, b, D, s
end
args = attractive_origin()
trajectories = sbtm(args...)
animation = animate_2d(trajectories, samples = 1:9)

# reproducing first numerical experiment: repelling particles attracted to a moving trap
function moving_trap()
    d = 2 # dimension of each particle
    N = 3 # number of particles in a sample
    a = 2. # trap amplitude
    w = 1. # trap frequency
    α = 0.5 # repelling force
    num_samples = 2 # number of samples
    Δts = 0.01*ones(10) # time increments
      
    # define drift vector field b and diffusion matrix D
    β(t) = a*[cos(π*w*t), sin(π*w*t)]
    function b(x, t)
        attract = β(t) .- x
        repel = α * (x .- mean(x, dims = 2))
        attract + repel
    end
    D(x, t) = 0.25
    
    # draw samples
    ρ₀ = MvNormal(β(0.), 0.25*I(d))
    xs = reshape(rand(ρ₀, N*num_samples), d, N, num_samples)

    s = Chain(
        Dense(d => 50, relu),
        Dense(50 => 50, relu),
        Dense(50 => 50, relu),
        Dense(50 => d))
    initialize_s!(s, ρ₀, xs)
    sbtm(xs, Δts, b, D, s)
end
trajectories = moving_trap()
runs = [trajectories[:,i,:]]
# TODO animate the result
# TODO switch to using Float32 as inputs
# TODO incorporate learning rate decay




# trying to initialize s to match the score
using Random: seed!
seed!(123)
d = 1 # dimension of each particle
N = 1 # number of particles in a sample
num_samples = 100 # number of samples
  
# draw samples
ρ₀ = MvNormal([0.], I(d))
xs = reshape(rand(ρ₀, N*num_samples), d, N, num_samples)

s = Chain(
#   Dense(d => 50, relu),
  Dense(d => d))
θ = params(s)
ys = gradient(x -> score(ρ₀, x), xs)[1]
ys_sum_squares = sum(ys.^2)
loss(s) = sum( (s(xs) - ys).^2 ) / ys_sum_squares
opt = Descent(10^-5)

loss(s)
for _ in 1:10^5
    grads = gradient(() -> loss(s), θ)
    Flux.update!(opt, θ, grads)
end
loss(s)



p=plot(xs[1,1,:], ys[1,1,:], marker = true, label = "actual");
plot!(p, xs[1,1,:], s(xs)[1,1,:], marker = true, label = "learned")

# Finding: immediately (after 1 step) get stuck in a local minimum