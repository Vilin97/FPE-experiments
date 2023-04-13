# benchmarking collision kernel for Landau 
using BenchmarkTools, LinearALgebra
z = xs[:,1]
a = zeros(eltype(z),3,3)
A1(z) = eltype(z)(1/24) * (norm(z)^2 * I(3) - z*z')
A2(z) = eltype(z)(1/24) .* (norm(z)^2 .* I(3) .- z*z')
A3(z) = eltype(z)(1/24) .* (sum(x -> x^2, z) .* I(3) .- z.*z')
A4(a, z) = a .= eltype(z)(1/24) .* (sum(x -> x^2, z) .* I3 .- z.*z')
A5(a, z) = a .= eltype(z)(1/24) .* (normsq(z) .* I3 .- z.*z')
A6(z) = eltype(z)(1/24) .* (normsq(z) .* I3 .- z.*z')

@btime A1($z)
@btime A2($z)
@btime A3($z)
@btime A4($a,$z)
@btime A5($a,$z)
@btime $a .= A6($z)


# benchmarking normsq
using BenchmarkTools
function normsq(z)
    res = zero(eltype(z))
    @fastmath for zi in z
        res += zi*zi
    end
    res
end

function normsq_lv(z)
    res = zero(eltype(z))
    @turbo for i in eachindex(z)
        res += z[i]*z[i]
    end
    res
end

z = rand(3)
@btime norm($z)^2 # 16.733 ns
@btime ($z)' * $z # 17.134 ns
@btime sum(x -> x^2, $z) # 4.400 ns
@btime sum(abs2, $z) # 4.400 ns
@btime normsq($z) # 3.600 ns
@btime normsq_lv($z) # 7.800 ns

z = rand(100)
@btime norm($z)^2 # 43.794 ns
@btime ($z)' * $z # 30.050 ns
@btime sum(x -> x^2, $z) # 14.314 ns
@btime sum(abs2, $z) # 14.314 ns
@btime normsq($z) # 10.200 ns
@btime normsq_lv($z) # 11.111 ns

# benchmarking dot product
using BenchmarkTools
function fastdot(z, v)
    res = zero(eltype(z))
    @fastmath for i in eachindex(z)
        res += z[i]*v[i]
    end
    res
end

function fastdot_lv(z, v)
    res = zero(eltype(z))
    @turbo for i in eachindex(z)
        res += z[i]*v[i]
    end
    res
end

z = rand(3)
v = rand(3)
@btime dot($z, $v) # 17.117 ns
@btime sum(x -> x[1]*x[2], zip($z, $v)) # 9.400 ns
@btime fastdot($z, $v) # 5.500 ns 
@btime fastdot_lv($z, $v) # 6.600 ns

z = rand(100)
v = rand(100)
@btime dot($z, $v) # 35.075 ns
@btime sum(x -> x[1]*x[2], zip($z, $v)) # 103.445 ns
@btime fastdot($z, $v) # 18.637 ns
@btime fastdot_lv($z, $v) # 14.414 ns 

# benchmarking landau_f!
include("src/utils.jl")
using Flux, LinearAlgebra, TimerOutputs, BenchmarkTools, LoopVectorization
const I3 = I(3)
num_particles(xs :: AbstractArray{T, 2}) where T = size(xs, 2)
"sum of squares of the elements of z"
function normsq(z)
    res = zero(eltype(z))
    @fastmath for zi in z
        res += zi*zi
    end
    res
end

function landau_f_test1!(dxs, xs, pars, t)
    A!, A_mat, s, s_values, temp1, temp2 = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    @timeit "l1" s_values .= s(xs)
    @timeit "propagate particles" @views for p in 1:n, q in 1:n
        temp1 .= xs[:,p] .- xs[:,q]
        A!(A_mat, temp1)
        temp1 .= s_values[:,q] .- s_values[:,p]
        mul!(temp2, A_mat, temp1)
        dxs[:,p] .+= temp2
    end
    dxs ./= n
    nothing
end

function landau_f_test2!(dxs, xs, pars, t)
    A!, A_mat, s, s_values, z, v = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    @timeit "l1" s_values .= s(xs)
    @timeit "propagate particles" @views for p in 1:n, q in 1:n
        z .= xs[:,p] .- xs[:,q]
        v .= s_values[:,q] .- s_values[:,p]
        dotzv = dot(z,v)
        normsqz = normsq(z)
        v .*= normsqz
        v .-= dotzv .* z
        dxs[:,p] .+= v ./ 24
    end
    dxs ./= n
    nothing
end

function landau_f_test3!(dxs, xs, pars, t)
    A!, A_mat, s, s_values, z, v = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    @timeit "l1" s_values .= s(xs)
    @timeit "propagate particles" @views for p in 1:n, q in 1:n
        z .= xs[:,p] .- xs[:,q]
        v .= s_values[:,q] .- s_values[:,p]
        dotzv = fastdot(z,v)
        normsqz = normsq(z)
        v .*= normsqz
        v .-= dotzv .* z
        dxs[:,p] .+= v
    end
    dxs ./= 24*n
    nothing
end

function landau_f_test4!(dxs, xs, pars, t)
    A!, A_mat, s, s_values, z, v = pars
    n = num_particles(xs)
    dxs .= zero(eltype(xs))
    @timeit "l1" s_values .= s(xs)
    @timeit "propagate particles" @views for p in 1:n, q in 1:n
        z .= xs[:,p] .- xs[:,q]
        v .= s_values[:,q] .- s_values[:,p]
        dxs[:,p] .+=  v .* normsq(z) .- fastdot(z,v) .* z
    end
    dxs ./= 24*n
    nothing
end

s = Chain(Dense(3, 100, relu), Dense(100, 3))
xs = randn(Float32, 3, 1000)
A!(a, z) = a .= eltype(z)(1/24) .* (normsq(z) .* I3 .- z.*z')
A_mat = zeros(eltype(xs), size(xs,1), size(xs,1))
s_values = similar(s(xs))
temp1 = similar(xs[:,1])
temp2 = similar(xs[:,1])
p = (A!, A_mat, s, s_values, temp1, temp2)
dxs = zeros(eltype(xs),size(xs))

reset_timer!()
landau_f_test3!(dxs, xs, p, 0.)
print_timer()

reset_timer!()
landau_f_test4!(dxs, xs, p, 0.)
print_timer()
@btime landau_f_test2!($dxs, $xs, $p, 0.)
@btime landau_f_test4!($dxs, $xs, $p, 0.)

# benchmarking landau_f!
n = 1000
seed!(1234)
xs, ts, A!, ρ = landau(n)
ρ₀ = x->ρ(x,0.)
s = initialize_s(ρ₀, xs, 100, 1)

A_mat = zeros(eltype(xs), size(xs,1), size(xs,1))
s_values = similar(s(xs))
diff = similar(xs[:,1])
p = (A!, A_mat, s, s_values, diff)

dxs = zeros(eltype(xs),size(xs))
function measure(dxs, xs, p, t)
    reset_timer!()
    landau_f_test!(dxs, xs, p, 0.)
    print_timer()
end
measure(dxs, xs, p, 0.)
@btime landau_f_test!($dxs, $xs, $p, 0.) # only s(xs) allocates

# benchmarking initialize_s
using BenchmarkTools, TimerOutputs, LinearAlgebra, Flux
using Random: seed!
using Distributions: MvNormal, logpdf
include("src/utils.jl")
include("src/sbtm.jl")

# plotting the approximation
start_time = 5.5
n = 10_000
seed!(1234)
xs, ts, ρ = landau(n, start_time)
ρ₀ = x -> ρ(x, 0.0)
reset_timer!()
@timeit "2-layer" s2 = initialize_s(ρ₀, xs, 100, 2; verbose = 2)
@timeit "3-layer" s3 = initialize_s(ρ₀, xs, 100, 3; verbose = 2)
print_timer()
# train error:
l2_error_normalized(s2, xs, score(ρ₀, xs))
l2_error_normalized(s3, xs, score(ρ₀, xs))
# test error: 
xs_new, ts, ρ = landau(n, start_time)
xs_high_n, ts, ρ = landau(10*n, start_time)
l2_error_normalized(s2, xs_new, ρ₀)
l2_error_normalized(s2, xs_high_n, ρ₀)
l2_error_normalized(s3, xs_new, ρ₀)
l2_error_normalized(s3, xs_high_n, ρ₀)

plt = plot(title = "score at t = $start_time, n = $n");
plot!(plt, -4:0.01:4, x -> score(ρ₀,[x,0.,0.])[1], label="true");
plot!(plt, -4:0.01:4, x -> s2([x,0.,0.])[1], label="s2");
plot!(plt, -4:0.01:4, x -> s3([x,0.,0.])[1], label="s3");
scatter!(plt, xs[1,:], score(ρ₀, vcat(xs[1,:]', zeros(2,n)))[1,:], label="samples", markersize=2);
plt


start_time = 5.5
n = 10_000
max_iterations = 2*10^4
seed!(1234)
xs, ts, ρ = landau(n, start_time)
ρ₀ = x -> ρ(x, 0.0)
reset_timer!()
@timeit "2-layer" s2 = initialize_s(ρ₀, xs, 100, 2; verbose = 2, max_iterations = max_iterations)
@timeit "3-layer" s3 = initialize_s(ρ₀, xs, 100, 3; verbose = 2, max_iterations = max_iterations)
print_timer()
# train error:
l2_error_normalized(s2, xs, score(ρ₀, xs))
l2_error_normalized(s3, xs, score(ρ₀, xs))
# test error: 
xs_new, ts, ρ = landau(n, start_time)
xs_high_n, ts, ρ = landau(10*n, start_time)
l2_error_normalized(s2, xs_new, ρ₀)
l2_error_normalized(s2, xs_high_n, ρ₀)
l2_error_normalized(s3, xs_new, ρ₀)
l2_error_normalized(s3, xs_high_n, ρ₀)

plt = plot(title = "score at t = $start_time, n = $n");
plot!(plt, -4:0.01:4, x -> score(ρ₀,[x,0.,0.])[1], label="true");
plot!(plt, -4:0.01:4, x -> s2([x,0.,0.])[1], label="s2");
plot!(plt, -4:0.01:4, x -> s3([x,0.,0.])[1], label="s3");
scatter!(plt, xs[1,:], score(ρ₀, vcat(xs[1,:]', zeros(2,n)))[1,:], label="samples", markersize=2);
plt

# comparing the error
function square_error(s, xs)
    ys =  score(ρ₀, xs)
    sum(abs2, s(xs) - ys) / sum(abs2, ys)
end
square_error(s, xs)

n = 100
seed!(1234)
xs, ts, ρ = landau(n, start_time)
square_error(s, xs)

# choosing the activation function. softsign is best
n = 400
max_epochs = 4*10^4
seed!(1234)
xs, ts, ρ = landau(n, start_time)
ρ₀ = x -> ρ(x, 0.0)
reset_timer!()
plots = []
for activation in [relu, selu, softsign]
    println("")
    @show activation
    @timeit "$activation" s = initialize_s(ρ₀, xs, 100, 1, verbose = 1, max_epochs = max_epochs, activation = activation)

    plt = plot(ylims=(-9,9), title = "n = $n, max_epochs = $max_epochs, activation = $activation");
    plot!(plt, -4:0.01:4, x -> score(ρ₀,[x,0.,0.])[1], label="true");
    plot!(plt, -4:0.01:4, x -> s([x,0.,0.])[1], label="s");
    scatter!(plt, xs[1,:], score(ρ₀, vcat(xs[1,:]', zeros(2,n)))[1,:], label="samples", markersize=3)
    push!(plots, plt)
end
println("\n\n\n\n\n\n")
print_timer()
plot(plots..., layout = (3,3), size = (2200,1800))

# choosing hyperparameters
n = 2000
max_epochs = 4*10^4
seed!(1234)
xs, ts, ρ = landau(n, start_time)
ρ₀ = x -> ρ(x, 0.0)
loss_plots = []
@show n
nns = [
Chain(
    Dense(3, 100, softsign), 
    Dense(100, 3)),
Chain(
    Dense(3, 100, softsign), 
    Dense(100, 100, softsign), 
    Dense(100, 3)), 
Chain(
    Dense(3, 100, softsign), 
    Dense(100, 100, softsign), 
    Dense(100, 100, softsign), 
    Dense(100, 3))]
losses_array = []
reset_timer!()
for (i,s) in enumerate(nns)
    @show s
    @timeit "NN $i" _, losses = initialize_s!(s, ρ₀, xs, verbose = 2, max_epochs = max_epochs, record_losses = true)
    push!(losses_array, losses)
end
print_timer()
plt = plot(size = (1200,900), title = "n = $n, max_epochs = $max_epochs", xscale = :log10, yscale = :log10, xlabel = "epoch", ylabel = "loss");
for i in 1:length(nns)
    plot!(plt, 1:max_epochs, losses_array[i], label = "nn $i")
end
plt


# Choosing architecture. Looks like 3 hidden layers is best. 1 layer is not rich enough. Also testing mini-batches.
using Plots
using BenchmarkTools, TimerOutputs, LinearAlgebra, Flux
using Random: seed!
using Distributions: MvNormal, logpdf
include("src/utils.jl")
include("src/sbtm.jl")
function square_error(s, xs)
    ys =  score(ρ₀, xs)
    sum(abs2, s(xs) .- ys) / sum(abs2, ys)
end
n = 20_000
start_time = 6.
max_iterations = 2*10^4
seed!(1234)
xs, ts, ρ = landau(n, start_time)
ρ₀ = x -> ρ(x, 0.0)
loss_plots = []
@show n
losses_array = []
xs_test = landau(50_000, start_time)[1]
test_error_plt = plot(xlims = (-4,4), ylims=(-9,9), title = "n = $n, max_iterations = $max_iterations");
plot!(test_error_plt, -4:0.01:4, x -> score(ρ₀,[x,0.,0.])[1], label="true");
reset_timer!()
for (i,batchsize) in enumerate(2 .^ (6:10))
    @show batchsize
    seed!(1234)
    s = Chain(
    Dense(3, 100, softsign), 
    Dense(100, 100, softsign), 
    Dense(100, 100, softsign), 
    Dense(100, 3))
    @timeit "batchsize $batchsize" _, losses = initialize_s_new!(s, ρ₀, xs, verbose = 2, max_iterations = max_iterations, record_losses = true, batchsize = batchsize)
    push!(losses_array, losses)

    plot!(test_error_plt, -2:0.01:2, x -> s([x,0.,0.])[1], label="batchsize $batchsize");
    @show square_error(s, xs_test)
    @show square_error(s, xs)
end
print_timer()
plt = plot(size = (1200,900), title = "n = $n, max_iterations = $max_iterations", xscale = :log10, yscale = :log10, xlabel = "iteration", ylabel = "loss");
for i in 1:length(losses_array)
    plot!(plt, 1:max_iterations, losses_array[i], label = "batchsize $(2 .^ (i+5))")
end
plt
plot(test_error_plt, size = (2000, 1500))



# testing how callback works
using DifferentialEquations, TimerOutputs, BenchmarkTools
ts = 0.0:0.1:1.0
affect!(integrator) = (DifferentialEquations.u_modified!(integrator, false)); nothing
cb = PresetTimeCallback(ts, affect!, save_positions=(false,false))
f!(du,u,p,t) = @timeit "f!" (du .= u)
prob = ODEProblem(f!, [1.0], (0.0,1.0))


reset_timer!()
solution1 = solve(prob, alg = Euler(), saveat=ts, tstops = ts)
print_timer() # f! ncalls = 11
reset_timer!()
solution2 = solve(prob, alg = Euler(), saveat=ts, tstops = ts, callback = cb)
print_timer() # f! ncalls = 21
reset_timer!()
solution2 = solve(prob, alg = Euler(), saveat=ts, dt = 0.1, callback = cb)
print_timer()

@btime solve($prob, alg = $(Euler()), tstops = $ts) #  5.750 μs (79 allocations: 5.75 KiB)
@btime solve($prob, alg = $(Euler()), tstops = $ts, callback = $cb) # 7.875 μs (83 allocations: 6.22 KiB)
@btime solve($prob, alg = $(Euler()), dt = 0.1, callback = $cb) 