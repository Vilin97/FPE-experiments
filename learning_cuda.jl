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


