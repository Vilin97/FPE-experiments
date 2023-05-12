N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x             # increment each element of y with the corresponding element of x
using Test
@test all(y .== 3.0f0)


function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)


function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

# @btime sequential_add!($y, $x)
# @btime parallel_add!($y, $x)


using CUDA

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

@btime add_broadcast!($y_d, $x_d)


function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

@btime bench_gpu1!($y_d, $x_d)


function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

@btime bench_gpu2!($y_d, $x_d)


function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

numblocks = ceil(Int, N/256)

fill!(y_d, 2)
@cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end
end
@btime bench_gpu3!($y_d, $x_d)

kernel = @cuda launch=false gpu_add3!(y_d, x_d)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

fill!(y_d, 2)
kernel(y_d, x_d; threads, blocks)
@test all(Array(y_d) .== 3.0f0)

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end
@btime bench_gpu4!($y_d, $x_d)


function gpu_add2_print!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @cuprintln("thread $index, block $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

@cuda threads=16 gpu_add2_print!(y_d, x_d)
synchronize()

using CUDA
num_particles(xs :: AbstractArray{T, 2}) where T = size(xs, 2)
rec_epsilon(T :: Type, n) = 2 * T(kde_bandwidth(n)^2)
rec_epsilon(n) = 2 * kde_bandwidth(n)^2
kde_bandwidth(n, d = 3) = n^(-1/(d+4)) / √2
CUDA.allowscalar(false) # disable e.g. a[1] where a is a CuArray

a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+6)/6)
true_pdf(x, K=K(0f0)) = (a(K) + b(K)*sum(abs2, x)) * sqrt(1/(2*(π*K))^-3) * exp(-sum(abs2, x)/(2K))
function KL_divergence(u, true_pdf)
    n = num_particles(u)
    ε = rec_epsilon(eltype(u), n)
    u_cols = eachcol(u)
    Z = sum(true_pdf, u_cols)
    sum(x -> log(reconstruct_pdf(ε, x, u)*Z/true_pdf(x)), u_cols) / n
end

function reconstruct_pdf(ε, x, xs)
    n = num_particles(xs)
    exp_sum = -sum(abs2, xs .- x, dims = 1) ./ ε .|> exp |> sum
    exp_sum / sqrt((π*ε)^length(x)) / n
end

using BenchmarkTools
n = 10^3
u = CUDA.rand(3, n)
@btime KL_divergence(u, true_pdf)
u_cpu = Array(u)
@btime KL_divergence(u_cpu, true_pdf) # 20 times faster on a CPU

using CUDA
x = CUDA.rand(3)
xs = CUDA.rand(3, 10)
reconstruct_pdf(1f0, x, xs)
sum(y -> sum(x+y), xs, dims = 2, init = zero(eltype(xs))) # error
x+xs[:,1] # works
sum(y -> sum(x), xs, init = zero(eltype(xs)))

f(x, xs) = sum(y -> sum(x), xs)

e = nothing
try 
    reconstruct_pdf(1f0, x, xs)
catch e
end

cx = rand(3)
cxs = rand(3, 10)
sum(y -> sum(y), cxs, dims = 1, init = 0.)

xs= rand(3, 10)
xs_cols = eachcol(xs)
@btime sum(helper, $xs_cols)

n = 10^6
x = CUDA.rand(3)
cx = Array(cx)
xs = CUDA.rand(3, n)
cxs = Array(xs)
@show n
@btime sum(exp.(-sum(abs2, $cxs .- $cx,dims=1)))
@btime sum(exp.(-sum(abs2, $xs .- $x,dims=1)))

xs = rand(3,10)
x = rand(3)
sum(abs2, xs .- x, dims = 1)
xs = CUDA.rand(3,10)
x = CUDA.rand(3)
sum(abs2, xs .- x, dims = 1)

@btime sum(exp.(-sum(abs2, xs .- x,dims=1)))
@btime sum(exp.(-sum(abs2, xs .- x,dims=1)))

@btime reconstruct_pdf(1f0, $x, $xs)
@btime reconstruct_pdf(1f0, $cx, $cxs)

# timing
# CUDA.@time ...

# L2 error. HCubature does not work on a GPU
using HCubature, CUDA
d = 3
ε = 1f-1
a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+6)/6)
true_pdf(x, K=K(0f0)) = (a(K) + b(K)*sum(abs2, x)) * sqrt(1/(2*(π*K))^-3) * exp(-sum(abs2, x)/(2K))

function reconstruct_pdf(ε, x, xs)
    n = size(xs, 2)
    exp_sum = -sum(abs2, xs .- x, dims = 1) ./ ε .|> exp |> sum
    exp_sum / sqrt((π*ε)^length(x)) / n
end

CUDA.allowscalar(false)
u = CUDA.rand(3, 10^5)
x = CUDA.ones(3)
xc = Array(x)
uc = Array(u)
CUDA.@time reconstruct_pdf(1f-1, x, u) # 391 CPU allocations
CUDA.@time reconstruct_pdf(1f-1, xc, uc) # 23 CPU allocations

CUDA.@time integrand, accuracy = hcubature(x -> reconstruct_pdf(1f-1, x, u), CuArray(fill(-3, 3)), CuArray(fill(3, 3)), maxevals = 10^5)
CUDA.@time integrand, accuracy = hcubature(x -> reconstruct_pdf(1f-1, x, uc), (fill(-3, 3)), (fill(3, 3)), maxevals = 10^4)


# NN training
using CUDA
CUDA.functional()
Flux.GPU_BACKEND

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
loss(x, y) # ~ 3


d = Dense(10 => 5, σ)
d = fmap(cu, d)
d.weight # CuArray
d(cu(rand(10))) # CuArray output

m = Chain(Dense(10 => 5, σ), Dense(5 => 2), softmax)
m = fmap(cu, m)
m(cu(rand(10)))


using Flux, CUDA
m = Dense(10, 5) |> gpu
Dense(10 => 5)      # 55 parameters

x = rand(10) |> gpu
m(x)

test_data = cu(rand(3, 160_000))



l2_error_normalized(s, xs, ρ) = l2_error_normalized(s, xs, score(ρ, xs))
l2_error_normalized(s, xs, ys :: AbstractArray) = sum(abs2, s(xs) .- ys) / sum(abs2, ys)
T = Float32
optimiser = Adam(10^-3); loss_tolerance = T(10^-4); verbose = 0; max_iterations = 10^3; record_losses = false; batchsize=min(2^8, num_particles(xs))
ρ₀ = MvNormal(I(3))
iteration = 1

using TimerOutputs
CUDA.allowscalar(false)
n = 2^17

function my_train(s, xs, ys, max_iterations, data_loader, loss_tolerance, state)
    iteration = 1
    while iteration < max_iterations
        for (x, y) in data_loader
            @timeit "compute grad" loss_value, grads = withgradient(s -> l2_error_normalized(s, x, y), s)
            current_loss = loss_value
            if iteration >= max_iterations
                break
            end
            iteration += 1
            @timeit "update" Flux.update!(state, s, grads[1])
        end
    end
    
end

reset_timer!()
# cpu
seed!(1234)
s = Chain(Dense(3 => 100, softsign), Dense(100 => 3))
xs = rand(T, 3, n)
ys = score(ρ₀, xs)
data_loader = DataLoader((data=xs, label=ys), batchsize=batchsize);
state = Flux.setup(optimiser, s)
sum(abs2, s(xs) .- ys)
@timeit "cpu" my_train(s, xs, ys, max_iterations, data_loader, loss_tolerance, state)
sum(abs2, s(xs) .- ys)

# gpu
seed!(1234)
s = Chain(Dense(3 => 100, softsign), Dense(100 => 3)) |> gpu
xs = rand(T, 3, n)
ys = score(ρ₀, xs)
data_loader = DataLoader((data=xs, label=ys), batchsize=batchsize) |> gpu;
state = Flux.setup(optimiser, s)
sum(abs2, s(xs |> gpu) .- (ys |> gpu))
@timeit "gpu" my_train(s, xs, ys, max_iterations, data_loader, loss_tolerance, state)
sum(abs2, s(xs |> gpu) .- (ys |> gpu))

print_timer()

using BenchmarkTools, Flux, CUDA
using Zygote: withgradient
optimiser = Adam(10^-3); 

#cpu
n = 2^8
s = Chain(Dense(3 => 100, softsign), Dense(100 => 100, softsign), Dense(100 => 3))
state = Flux.setup(optimiser, s)
x = rand(Float32, 3, n)
y = rand(Float32, 3, n)
@btime _, grads = withgradient(s -> sum(abs2, s(x) .- y), $s) # 1.2 ms

#gpu
s = Chain(Dense(3 => 100, softsign), Dense(100 => 100, softsign), Dense(100 => 3)) |> gpu
state = Flux.setup(optimiser, s)
x = rand(3, n) |> gpu
y = rand(3, n) |> gpu
@btime CUDA.@sync _, grads = withgradient(s -> sum(abs2, s(x) .- y), s) # 986 μs

#simple chains
using SimpleChains
x = rand(Float32, 3, n)
y = rand(Float32, 3, n)
s = SimpleChain(
  static(3),
  TurboDense{true}(softsign, 100),
  TurboDense{true}(softsign, 100),
  TurboDense{true}(softsign, 3),
  SquaredLoss(y)
);
p = SimpleChains.init_params(s)
g = similar(p)
@btime valgrad!($g, $s, $x, $p) # 488 μs
