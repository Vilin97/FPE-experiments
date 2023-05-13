# implementing the algorithm from "probability flow solution of the fokker-planck equation" 2022

using Flux, LinearAlgebra, DifferentialEquations, LoopVectorization, TimerOutputs
using Distributions: MvNormal, logpdf
using Zygote: gradient, withgradient
using Flux.Optimise: Adam
using Flux: params
using MLUtils: DataLoader

function initialize_s(ρ₀, xs, size_hidden, num_hidden; activation = softsign, verbose = 0, kwargs...)
    d_bar = size(xs,1)
    s = Chain(
        Dense(d_bar => size_hidden, activation),
        repeat([Dense(size_hidden, size_hidden, activation)], num_hidden-1)...,
        Dense(size_hidden => d_bar)
        )
    @timeit "NN init" initialize_s!(s, ρ₀, xs; verbose = verbose, kwargs...)
    return s
end

l2_error_normalized(s, xs, ρ) = l2_error_normalized(s, xs, score(ρ, xs))
l2_error_normalized(s, xs, ys :: AbstractArray) = sum(abs2, s(xs) .- ys) / sum(abs2, ys)
function initialize_s!(s, ρ₀, xs :: AbstractArray{T}; optimiser = Adam(10^-3), loss_tolerance = T(10^-4), verbose = 0, max_iterations = 10^5, record_losses = false, batchsize=min(2^8, num_particles(xs)), kwargs...) where T
    verbose > 0 && println("Initializing NN for $(num_particles(xs)) particles.")
    verbose > 1 && println("Batch size = $batchsize, loss_tolerance = $loss_tolerance, max_iterations = $max_iterations. \n$s")
    ys = score(ρ₀, xs)
    data_loader = DataLoader((data=xs, label=ys), batchsize=batchsize);
    state = Flux.setup(optimiser, s)
    current_loss = l2_error_normalized(s, xs, ys)
    iteration = 1
    epoch = 1
    while iteration < max_iterations
        for (x, y) in data_loader
            loss_value, grads = withgradient(s -> l2_error_normalized(s, x, y), s)
            current_loss = loss_value
            if iteration >= max_iterations
                break
            end
            verbose > 2 && iteration % 1000 == 0 && println("Iteration $iteration, batch loss $current_loss")
            iteration += 1
            Flux.update!(state, s, grads[1])
        end
        loss = l2_error_normalized(s, xs, ys)
        verbose > 1 && epoch % 100 == 0 && println("Epoch $epoch, loss $loss")
        if loss < loss_tolerance
            break
        end
        epoch += 1
    end
    final_loss = l2_error_normalized(s, xs, ys)
    verbose > 0 && println("Initialized NN in $iteration iterations. Loss = $final_loss.")
    # iteration, losses
    nothing
end

function loss(s, xs :: AbstractArray{T}, α) where T
    ζ = randn(T, size(xs))
    denoise_val = ( s(xs .+ α .* ζ) ⋅ ζ - s(xs .- α .* ζ) ⋅ ζ ) / α
    (norm(s(xs))^2 + denoise_val)/num_particles(xs)
end

