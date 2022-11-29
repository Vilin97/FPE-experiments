# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, Plots
using Distributions: MvNormal, logpdf
using Zygote: gradient, pullback
using Flux.Optimise: Adam
using Flux: params, train!
using Statistics: mean

using Flux.OneHotArrays: onehot

# divergence of vector field s at x
function divergence(f, v)
    _, ∂f = pullback(f, v)
    sum(eachindex(v)) do i
        ∂fᵢ = ∂f(onehot(i, eachindex(v)))
        sum(x -> x[i], ∂fᵢ)
    end
end
# divergence(s, x) = sum(gradient(x -> s(x)[i], x)[1][i] for i in 1:length(x))
loss(s, xs :: Array{T, 3}) where T =  (sum(s(xs).^2) + T(2.0)*divergence(s, xs))/size(xs, 3)

score(ρ, x) = convert(eltype(x), sum(logpdf(ρ, x)))
propagate(x, t, Δt, b, D, s) = x + Δt * b(x, t) + D(x, t)*s(x)

function plot_losses(losses, epochs)
    p = plot(vec(losses), title = "Score approximation", xaxis = "epochs", yaxis = "loss", label = "training loss")
    scatter!(p, 1:epochs:length(vec(losses)), vec(losses)[1:epochs:end], label = "discrete time propagation", marker = true)
end

function custom_train!(s, xs; optimiser = Adam(5. * 10^-3), epochs = 25, record_losses = true)
    θ = params(s)
    losses = zeros(typeof(loss(s, xs)), epochs)
    for epoch in 1:epochs
        record_losses && (losses[epoch] = loss(s, xs))
        @show epoch, losses[epoch]
        grads = gradient(() -> loss(s, xs), θ)
        Flux.update!(optimiser, θ, grads)
    end
    losses
end

function initialize_s!(s, ρ₀, xs; optimiser = Adam(10^-4), ε = 10^-4)
    ys = gradient(x -> score(ρ₀, x), xs)[1]
    ys_sum_squares = sum(ys.^2)
    square_error(s) = sum( (s(xs) - ys).^2 / ys_sum_squares )
    epoch = 0
    θ = params(s)
    while square_error(s) > ε
        grads = gradient(() -> square_error(s), θ)
        Flux.update!(optimiser, θ, grads)
        epoch += 1
    end
    epoch
end

"""
xs  : sample from initial probability distribution
Δts : list of Δt's
b   : Rᵈ → Rᵈ
D   : Rᵈ → Rᵈˣᵈ
n   : number of particles
s   : NN to approximate score ∇log ρ
"""
function sbtm(xs, Δts, b, D, s; kwargs...)
    trajectories = zeros(eltype(xs), size(xs)..., 1+length(Δts)) # trajectories[:, i, j, k] is particle i of sample j at time k
    trajectories[:, :, :, 1] = xs
    losses = sbtm!(trajectories, Δts, b, D, s; kwargs...)
    trajectories, losses
end

function sbtm!(trajectories :: Array{T, 4}, Δts, b, D, s; epochs = 25, kwargs...) where T
    t = zero(eltype(Δts))
    losses = zeros(T, epochs, length(Δts))
    for (k, Δt) in enumerate(Δts)
        @show k
        xs_k = trajectories[:, :, :, k]
        losses[:, k] = custom_train!(s, xs_k; kwargs...)
        @show k
        for (j, x) in enumerate(eachslice(xs_k, dims = 3))
            trajectories[:, :, j, k+1] = propagate(x, t, Δt, b, D, s)
        end
        t += Δt
    end
    losses
end

# learning to animate
function animate_2d(trajectories; samples = 1, fps = 5, target = nothing)
    xmax = maximum(abs.(trajectories[1,:,samples,:])) + 0.5
    ymax = maximum(abs.(trajectories[2,:,samples,:])) + 0.5
    p = scatter(trajectories[1,:,samples,1], trajectories[2,:,samples,1], 
                title = "time 0",
                label = nothing, 
                color=RGB(0.0, 0.0, 1.0), 
                xlims = (-xmax, xmax), ylims = (-ymax, ymax),
                markersize = 3);
    target !== nothing && scatter!(p, [target[1, 1]], [target[2, 1]], markershape = :star, label = "target", color = :red)
    
    anim = @animate for k ∈ axes(trajectories, 4)
        λ = k/size(trajectories, 4)
        red = λ > 0.5 ? 2. *(λ - 0.5) : 0.
        green = 1. - abs(1. - 2. * λ)
        blue = λ < 0.5 ? 2. * (0.5-λ) : 0.
        target !== nothing && scatter!(p, [target[1, k]], [target[2, k]], markershape = :star, color = :red, label = nothing)
        scatter!(p, vec(trajectories[1,:,samples,k]), vec(trajectories[2,:,samples,k]), 
                  title = "time $k",
                  label = nothing, 
                  color = RGB(red, green, blue), 
                  xlims = (-xmax, xmax), ylims = (-ymax, ymax),
                  markersize = 3)
  end
  gif(anim, "anim_fps$fps.gif", fps = fps)
end

# experimenting with all particles attracting to the origin
function attractive_origin()
    d = 2
    N = 1
    num_samples = 9
    Δts = 0.01*ones(Float32, 5)
    b(x, t) = -x
    D(x, t) = Float32(0.1)
    xs = reshape(Float32[-1  0  1 -1  0  1 -1  0  1;
    -1 -1 -1  0  0  0  1  1  1], d, N, num_samples);
    s = Chain(
      Dense(d => 50, relu),
      Dense(50 => d))
    print("Initializing s...")
    initialize_s!(s, MvNormal(zeros(d), I(d)), xs, ε = 10^-2)
    xs, Δts, b, D, s
end
args = attractive_origin()
trajectories, losses = sbtm(args...)
plt = plot_losses(losses, 25)
animation = animate_2d(trajectories, samples = 1:9)

# reproducing first numerical experiment: repelling particles attracted to a moving trap
function moving_trap()
    d = 2 # dimension of each particle
    N = 3 # number of particles in a sample
    a = Float32(2.) # trap amplitude
    w = Float32(1.) # trap frequency
    α = Float32(0.5) # repelling force
    num_samples = 2 # number of samples
    Δts = 0.01*ones(Float32, 100) # time increments
      
    # define drift vector field b and diffusion matrix D
    β(t) = a*Float32[cos(π*w*t), sin(π*w*t)]
    function b(x, t)
        attract = β(t) .- x
        repel = α * (x .- mean(x, dims = 2))
        attract + repel
    end
    D(x, t) = Float32(0.25)
    
    # draw samples
    ρ₀ = MvNormal(β(0.), 0.25*I(d))
    xs = convert(Array{Float32, 3}, reshape(rand(ρ₀, N*num_samples), d, N, num_samples))

    h = 32
    s = Chain(
        Dense(d => h, relu),
        Dense(h => h, relu),
        Dense(h => h, relu),
        Dense(h => d))
    epochs = initialize_s!(s, ρ₀, xs)
    print("Took $epochs epochs to initialize. Initial loss: $(loss(s,xs))")
    xs, Δts, b, D, s
end
args = moving_trap()
trajectories, losses = sbtm(args...; optimiser = Adam(10^-3), record_losses = true)
animation = animate_2d(trajectories, samples = 1, fps = 1)
p = plot(vec(losses), title = "losses", xaxis = "epochs")
plot!(p, vec(losses)[1:25:end], label = nothing, marker = true)

# TODO animate the trap as well
# TODO do not redefine the anonymous functions every time in custom_train!, divergence, etc
# TODO can I propagate all xs at once?
# TODO double-check b(x,t) in moving_trap
# TODO figure out loss blowing up in the stbm