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
using BenchmarkTools, LoopVectorization, LinearAlgebra
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

normsq_sum(z) = sum(i -> z[i]^2, eachindex(z))

z = rand(3)
@btime norm($z)^2 # 16.733 ns
@btime ($z)' * $z # 17.134 ns
@btime normsq_sum($z) # 10 ns
@btime normsq_lv($z) # 7.800 ns
@btime sum(x -> x^2, $z) # 4.400 ns
@btime sum(abs2, $z) # 4.400 ns
@btime normsq($z) # 3.600 ns

z = rand(100)
@btime norm($z)^2 # 43.794 ns
@btime ($z)' * $z # 30.050 ns
@btime normsq_sum($z) # 26 ns
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
    @timeit "propagate particles" @turbo for p = 1:n
        Base.Cartesian.@nexprs 3 i -> dx_i = zero(eltype(dxs))
        for q = 1:n
            dotzv = zero(eltype(dxs))
            normsqz = zero(eltype(dxs))
            Base.Cartesian.@nexprs 3 i -> begin
                z_i = xs[i, p] - xs[i, q]
                v_i = s_values[i, q] - s_values[i, p]
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

n = 10^4
s = Chain(Dense(3, 100, relu), Dense(100, 3))
xs = randn(Float32, 3, n)
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

dxs_cp = deepcopy(dxs)

reset_timer!()
landau_f_test4!(dxs, xs, p, 0.)
print_timer()

dxs ≈ dxs_cp

@btime landau_f_test2!($dxs, $xs, $p, 0.)
@btime landau_f_test3!($dxs, $xs, $p, 0.)
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
using DifferentialEquations, TimerOutputs
ts = 0.0:0.1:1.0
affect!(integrator) = (DifferentialEquations.u_modified!(integrator, false)); nothing
cb = PresetTimeCallback(ts .+ 1e-6, affect!, save_positions=(false,false))
function f!(du,u,p,t) 
    @timeit "f!" (du .= u)
end
prob = ODEProblem(f!, [1.0], (0.0,1.0))

reset_timer!()
solution1 = solve(prob, alg = Euler(), saveat=ts, tstops = ts)
print_timer() # f! ncalls = 11
reset_timer!()
solution2 = solve(prob, alg = Euler(), saveat=ts, tstops = ts, callback = cb)
print_timer() # f! ncalls = 21
reset_timer!()
solution3 = solve(prob, alg = Euler(), saveat=ts, dt = 0.1, callback = cb)
print_timer() # f! ncalls = 12
reset_timer!()
solution4 = solve(prob, alg = Euler(), saveat=ts, dt = 0.1, tstops = ts, callback = cb)
print_timer() # f! ncalls = 12

maximum(abs, vcat((solution1.u .- solution3.u)...)) # 2.36e-7
maximum(abs, vcat((solution1.u .- solution2.u)...)) # 2.36e-6
maximum(abs, vcat((solution2.u .- solution3.u)...)) # 2.12e-6

@btime solve($prob, alg = $(Euler()), tstops = $ts) #  5.750 μs (79 allocations: 5.75 KiB)
@btime solve($prob, alg = $(Euler()), tstops = $ts, callback = $cb) # 7.875 μs (83 allocations: 6.22 KiB)
@btime solve($prob, alg = $(Euler()), dt = 0.1, callback = $cb) 


# Solving Landau with blob
n = 4_000
time_interval = 0.5
seed!(1234)
num_runs = 5
xs, ts, _ = landau(n, t_start; time_interval = time_interval)

solutions = []
εs = [0.005, 0.01, 0.025]
# εs = [0.025]
for ε in εs
    seed!(1234)
    avg_solution = blob_landau(xs, ts; ε = ε, saveat = ts[[1, end]], verbose = 1).u
    for run in 1:num_runs-1
        xs, ts, _ = landau(n, t_start; time_interval = time_interval)
        solution = blob_landau(xs, ts; ε = ε, saveat = ts[[1, end]], verbose = 1)
        avg_solution = hcat.(avg_solution, solution.u)
    end
    push!(solutions, avg_solution)
end
plotly()
x_range = -4:0.01:4
reconstruction_ε = 0.1
plt = plot(title = "slice comparison, n = $n, reconstruction_ε = $reconstruction_ε, num_runs = $num_runs", ylabel = "density", xlabel = "x", size = (1000, 600));
# plot!(plt, x_range, x -> f([x,0,0], K(0)), label="analytic, t=$start_time");
plot!(plt, x_range, x -> f([x,0,0], K(time_interval)), label="analytic, t=$(start_time+time_interval)");
# plot!(plt, x_range, x -> reconstruct_pdf(reconstruction_ε, [x,0,0], xs), label="sampled, t = $start_time")
for (solution, ε) in zip(solutions, εs)
    l2_error = Lp_error(solution[end], reconstruction_ε, marginal_pdf = x -> f(x, K(time_interval)), k=3, verbose = 2)
    plot!(plt, x_range, x -> reconstruct_pdf(reconstruction_ε, [x,0,0], solution[end]), label="solved, t = $(start_time+time_interval), ε = $(round(ε, digits=4)), l2 error = $l2_error")
end
plt

Lp_error(solutions[2][1], 0.1, marginal_pdf = x -> f(x, K(0.5)), k=3, verbose = 2)
Lp_error(solutions[2][end], 0.1, marginal_pdf = x -> f(x, K(0.5)), k=3, verbose = 2)

# training and saving a model
using JLD2
start_time = 6
n = 4_000
xs, ts, ρ = landau(n, start_time)
K(t) = 1 - exp(-(t+start_time)/6)
ρ₀ = x -> ρ(x, K(0))

s = initialize_s(ρ₀, xs, 100, 2; verbose = 2)
xs_big, ts, ρ = landau(20*n, start_time)
l2_loss = l2_error_normalized(s, xs_big, ρ₀)

s1 = initialize_s(ρ₀, xs, 100, 1; verbose = 2)
l2_loss1 = l2_error_normalized(s1, xs_big, ρ₀)
save("models/landau_model_n_$(n)_start_$start_time.jld2", "s", s, "l2_loss", l2_loss)
s_loaded = load("models/landau_model_n_$(n)_start_$start_time.jld2", "s")
s_loaded(xs_big) == s(xs_big)

plt = plot(title = "score at t = $start_time, n = $n");
plot!(plt, -4:0.01:4, x -> score(ρ₀,[x,0.,0.])[1], label="true");
plot!(plt, -4:0.01:4, x -> s([x,0.,0.])[1], label="two hidden layers, l2 loss = $l2_loss");
plot!(plt, -4:0.01:4, x -> s1([x,0.,0.])[1], label="one hidden layer, l2 loss = $l2_loss1");
plt

# kernel density estimation
using KernelDensity
kde([0.])

# benchmarking particle propagation
using BenchmarkTools
using LinearAlgebra
using StaticArrays
using LoopVectorization

function fun1!(dxs, xs, s_values)
    n = size(dxs, 2)
    z = zeros(3)
    v = zeros(3)
    @views for p = 1:n, q = 1:n
        z .= xs[:, p] .- xs[:, q]
        v .= s_values[:, q] .- s_values[:, p]
        dotzv = dot(z, v)
        normsqz = sum(abs2, z)
        v .*= normsqz
        v .-= dotzv .* z
        dxs[:, p] .+= v
    end
end

function normsq(z)
    res = zero(eltype(z))
    @fastmath for zi in z
        res += zi * zi
    end
    res
end

function fastdot(z, v)
    res = zero(eltype(z))
    @fastmath for i in eachindex(z)
        res += z[i] * v[i]
    end
    res
end

function fun2!(dxs, xs, s_values)   # with fastdot etc
    n = size(dxs, 2)
    z = zeros(3)
    v = zeros(3)
    @views for p = 1:n, q = 1:n
        z .= xs[:, p] .- xs[:, q]
        v .= s_values[:, q] .- s_values[:, p]
        dotzv = fastdot(z, v)
        normsqz = normsq(z)
        v .*= normsqz
        v .-= dotzv .* z
        dxs[:, p] .+= v
    end
end

function fun3!(dxs, xs, s_values)   # with fused broadcast
    n = size(dxs, 2)
    z = zeros(3)
    v = zeros(3)
    @views for p = 1:n, q = 1:n
        z .= xs[:, p] .- xs[:, q]
        v .= s_values[:, q] .- s_values[:, p]
        dotzv = fastdot(z, v)
        normsqz = normsq(z)
        dxs[:, p] .+= v .* normsqz .- dotzv .* z
    end
end

function fun3b!(Dxs, Xs, S_values)
    n = size(Dxs, 2)
    z = @SVector zeros(3)
    v = @SVector zeros(3)

    dxs = reinterpret(SVector{3,Float64}, Dxs)
    xs = reinterpret(SVector{3,Float64}, Xs)
    s_values = reinterpret(SVector{3,Float64}, S_values)

    for p = 1:n, q = 1:n
        z = xs[p] - xs[q]
        v = s_values[q] - s_values[p]
        dotzv = dot(z, v)
        normsqz = dot(z, z)
        dxs[p] += v * normsqz - dotzv * z
    end
end
function fun3c!(dxs, xs, s_values)
    n = size(dxs, 2)
    z = @MVector zeros(3)
    v = @MVector zeros(3)

    @views for p = 1:n, q = 1:n
        z .= xs[:, p] .- xs[:, q]
        v .= s_values[:, q] .- s_values[:, p]
        dotzv = dot(z, v)
        normsqz = sum(abs2, z)
        dxs[:, p] .+= v .* normsqz .- dotzv .* z
    end
    return
end
@fastmath function fun_cartesian!(dxs, xs, s_values)
    n = size(dxs, 2)
    @inbounds for p = 1:n
        Base.Cartesian.@nexprs 3 i -> dx_i = zero(eltype(dxs))
        for q = 1:n
            dotzv = zero(eltype(dxs))
            normsqz = zero(eltype(dxs))
            Base.Cartesian.@nexprs 3 i -> begin
                z_i = xs[i, p] - xs[i, q]
                v_i = s_values[i, q] - s_values[i, p]
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
    return
end
function fun_lv!(dxs, xs, s_values)
    n = size(dxs, 2)
    @turbo for p = 1:n
        Base.Cartesian.@nexprs 3 i -> dx_i = zero(eltype(dxs))
        for q = 1:n
            dotzv = zero(eltype(dxs))
            normsqz = zero(eltype(dxs))
            Base.Cartesian.@nexprs 3 i -> begin
                z_i = xs[i, p] - xs[i, q]
                v_i = s_values[i, q] - s_values[i, p]
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
    return
end

n = 1000
xs = rand(3, n)
s_values = rand(3, n)
dxs = rand(3, n)
dxs_2 = copy(dxs)
dxs_3 = copy(dxs)
fun1!(dxs_2, xs, s_values)
fun2!(dxs_3, xs, s_values)
@assert dxs_2 ≈ dxs_3
copyto!(dxs_3, dxs)
fun3!(dxs_3, xs, s_values)
@assert dxs_2 ≈ dxs_3
copyto!(dxs_3, dxs)
fun3b!(dxs_3, xs, s_values)
@assert dxs_2 ≈ dxs_3
copyto!(dxs_3, dxs)
fun3c!(dxs_3, xs, s_values)
@assert dxs_2 ≈ dxs_3
copyto!(dxs_3, dxs)
fun_cartesian!(dxs_3, xs, s_values)
@assert dxs_2 ≈ dxs_3
copyto!(dxs_3, dxs)
fun_lv!(dxs_3, xs, s_values)
@assert dxs_2 ≈ dxs_3
dxs_t = permutedims(dxs)'
xs_t = permutedims(xs)'
s_values_t = permutedims(s_values)'
fun_lv!(dxs_t, xs_t, s_values_t)
@assert dxs_2 ≈ dxs_t

@btime fun1!($dxs, $xs, $s_values)
@btime fun2!($dxs, $xs, $s_values)
@btime fun3!($dxs, $xs, $s_values)
@btime fun3b!($dxs, $xs, $s_values)
@btime fun3c!($dxs, $xs, $s_values)
@btime fun_cartesian!($dxs, $xs, $s_values)
@btime fun_cartesian!($dxs_t, $xs_t, $s_values_t)
@btime fun_lv!($dxs, $xs, $s_values)
@btime fun_lv!($dxs_t, $xs_t, $s_values_t)

# tuning learning rate for Landau
using JLD2, TimerOutputs, Plots
include("src/utils.jl")
include("src/sbtm.jl")
include("src/landau/sbtm.jl")
include("src/blob.jl")
include("src/landau/blob.jl")
include("src/plotting_utils.jl")

const START = 6
a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+START)/6)
true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * (2π*K)^(-3/2) * exp(-sum(abs2, x)/(2K))
true_slice(x, K) = true_pdf([x,0,0], K)
K(t) = 1 - exp(-(t+start_time)/6)

start_time = 6
pre_trained_s = load("models/landau_model_n_4000_start_6.jld2", "s")

n = 10_000
xs, ts, ρ = landau(n, start_time)
saveat = ts
s_ = deepcopy(pre_trained_s)
@timeit "initialize NN" initialize_s!(s_, x -> ρ(x,K(0)), xs, loss_tolerance = 5e-4, verbose = 2, max_iter = 10^4)

lrs = 10 .^ (-6:0.5:-3)
sols = []
for lr in lrs
    s = deepcopy(s_)
    solution, s_values, losses = sbtm_landau(xs, ts; s = s, verbose = 2, saveat = saveat, record_s_values = true, record_losses = true, optimiser = Adam(lr))
    push!(sols, (solution, s_values, losses))
end

plotly()
plt = plot(title = "landau NN error, start $START, ∑ᵢ|s(Xᵢ) - ∇log ρ(Xᵢ)|^2 / ∑ᵢ|∇log ρ(Xᵢ)|^2", xlabel = "time", ylabel = "error", size = (1000, 600));
for (i,lr) in enumerate(lrs)
    solution, s_values, _ = sols[i]

    errors = zeros(length(ts))
    @views for (k,t) in enumerate(ts)
        xs = solution[k]
        ys = score(x -> true_pdf(x, K(t)), xs)
        errors[k] = sum(abs2, s_values[:,:,k] .- ys) / sum(abs2, ys)
    end
    plot!(plt, ts, errors, label = "lr = $lr", marker = :circle, markersize = 3, linewidth = 2)
end
pdf_plt = pdf_plot([sol for (sol, _, _) in sols], ["lr $lr" for lr in lrs], rec_epsilon(n), 1, ts);
plot(pdf_plt, size = (1000, 600))
losses_plt = plot();
for (i,lr) in enumerate(lrs)
    _, _, losses = sols[i]
    plot_losses!(losses_plt, losses, label = "lr = $lr")
end
losses_plt
plt
losses_plt
plot(plt, pdf_plt, layout = (2,1), size = (1600, 1000))

s = deepcopy(s_)
reset_timer!()
solution, s_values, losses = sbtm_landau(xs, ts; s = s, verbose = 2, saveat = saveat, record_s_values = true)
print_timer()
solution[1] == xs
t = ts[1]
ys = score(x -> true_pdf(x, K(t)), solution[1])
sum(abs2, s_values[:,:,1] .- ys) / sum(abs2, ys)
sum(abs2, s_(xs) .- ys) / sum(abs2, ys)

maximum(abs, solution[1] .- solution[2])

# hyperparameter sweep for sbtm, landau
start_time = 6
n = 5_000
pre_trained_s = load("models/landau_model_n_4000_start_6.jld2", "s")
K(t) = 1 - exp(-(t+start_time)/6)
# seed!(1)
xs, ts, ρ = landau(n, start_time)
ρ₀(x) = ρ(x,K(0))
saveat = [0.0, 0.01, 0.25, 0.5]
s_ = deepcopy(pre_trained_s)
@timeit "initialize NN" initialize_s!(s_, ρ₀, xs, loss_tolerance = 1e-4, verbose = 1, max_iter = 10^4)

# sweep over learning rates. 10^-4 is best
reset_timer!()
print("lr sweep")
lrs = [10^-5, 10^-4, 10^-3, 10^-2]
for lr in lrs
    sbtm_landau(xs, ts; s = s_, verbose = 1, saveat = saveat, record_s_values = true, optimiser = Adam(lr))
end
print_timer()

gr()
plots = []
for t in [0.0, 0.01, 0.25, 0.5]
    plt = plot()
    plot!(plt,-2:0.01:2, x -> score(z -> true_pdf(z, K(t)), [x,0,0])[1], label = "true score at t = $t")
    for lr in lrs
        s = JLD2.load("models/landau_model_t_$(t)_n_$(n)_lr_$(lr)_α_$(0.4)_epochs_$(25)_layers_$(4).jld2", "s")
        plot!(plt,-2:0.01:2, x -> s([x,0,0])[1], label = "learning rate = $(round(lr, digits=7)), t = $t");
    end
    push!(plots, plt)
end
plot(plots..., layout = (2, 2), size = (1800, 1300), title = "score NN")

# sweep over denoising alphas. 0.4 is best
reset_timer!()
println("denoising alpha sweep")
alphas = [1e-3, 1e-2, 0.1, 0.5, 1.]
models_array = Matrix{Any}(undef, length(saveat), length(alphas))
solutions = []
for (j,alpha) in enumerate(alphas)
    solution, models, _ = sbtm_landau(xs, ts; s = s_, verbose = 2, saveat = saveat, record_models = true, denoising_alpha = alpha)
    models_array[:, j] .= models
    push!(solutions, solution)
end
print_timer()

gr()
plots = []
for (i,t) in enumerate(saveat)
    plt = plot()
    plot!(plt,-2:0.01:2, x -> score(z -> true_pdf(z, K(t)), [x,0,0])[1], label = "true score at t = $t")
    for (j,alpha) in enumerate(alphas)
        s = models_array[i,j]
        u = solutions[j][i]
        model_l2_error = (sum(abs2, s(u) - score(z -> true_pdf(z, K(t)), u)) / n)^(1/2)
        println("t = $t, alpha = $alpha, error = $model_l2_error")
        
        plot!(plt,-2:0.01:2, x -> s([x,0,0])[1], label = "denoising alpha = $(round(alpha, digits=7)), t = $t, error = $model_l2_error");
    end
    push!(plots, plt)
end
plot(plots..., layout = (2, 2), size = (1800, 1300), title = "score NN")

# sweep over epochs
reset_timer!()
print("epoch sweep")
epoch_nums = [10, 25, 50, 100]
for epochs in epoch_nums
    sbtm_landau(xs, ts; s = s_, verbose = 2, saveat = saveat, record_s_values = true, epochs = epochs)
end
print_timer()

gr()
plots = []
for t in [0.0, 0.01, 0.25, 0.5]
    plt = plot()
    plot!(plt,-2:0.01:2, x -> score(z -> true_pdf(z, K(t)), [x,0,0])[1], label = "true score at t = $t")
    for epochs in epoch_nums
        s = JLD2.load("models/landau_model_t_$(t)_n_$(n)_lr_$(10^-4)_α_$(0.4)_epochs_$(epochs)_layers_$(4).jld2", "s")
        plot!(plt,-2:0.01:2, x -> s([x,0,0])[1], label = "epochs = $epochs, t = $t");
    end
    push!(plots, plt)
end
plot(plots..., layout = (2, 2), size = (1800, 1300), title = "score NN")



# evluating the denoising trick 
using Zygote, Flux, JLD2, LinearAlgebra, OneHotArrays, Plots
function divergence(f, v)
    _, ∂f = pullback(f, v)
    sum(eachindex(v)) do i
        ∂fᵢ = ∂f(onehot(i, eachindex(v)))
        sum(x -> x[i], ∂fᵢ)
    end
end
function denoise(s, x :: AbstractVector{T}, α, n) where T
    ζs = randn(T, length(x), n)
    sh = s(x .+ α .* ζs)
    sl = s(x .- α .* ζs)
    denoise_vals = [(sh_val ⋅ ζ - sl_val ⋅ ζ)/(2α) for (sh_val, sl_val, ζ) in zip(eachcol(sh), eachcol(sl), eachcol(ζs))]
end
rec_epsilon(n, d = 3) = 2 * kde_bandwidth(n, d)^2
kde_bandwidth(n, d = 3) = n^(-1/(d+4)) / √2
num_particles(xs :: AbstractArray{T, 1}) where T = size(xs, 1)
mol(ε, x) = exp(-sum(abs2, x)/ε)/sqrt((π*ε)^length(x)) 
Mol(ε, x, xs :: AbstractVector) = sum( mol(ε, x - x_q) for x_q in xs )
reconstruct_pdf(ε, x, u :: AbstractVector) = Mol(ε, x, u)/length(u)
reconstruct_pdf(x, u) = reconstruct_pdf(rec_epsilon(num_particles(u)), x, u)

s = load("models/landau_model_n_4000_start_6.jld2", "s")
N = 10^4
alphas = Float32[0.1, 0.3, 0.5, 1.]
plotting_colors = [:red, :blue, :green, :orange]
gr()

plots = []
for x in [Float32[r,0,0] for r in [0., 0.5, 1, 1.5, 2, 2.5]]
    true_div = divergence(s, x)
    plt = plot(title = "denoising trick at x = $(x[1])", size = (1600, 1200), xlabel = "position x", ylabel = "pdf of s(x + αζ)⋅ζ/α)");
    for (i,alpha) in enumerate(alphas)
        vals = denoise(s, x, alpha, N)
        μ = sum(vals)/N
        bias = μ - true_div
        var = sum(abs2, vals .- μ) / (N-1)
        expected_mse = bias^2 + var
        @show alpha, μ, bias, var, expected_mse
        plot!(plt, true_div .+ (-10:0.1:10), x_ -> reconstruct_pdf(rec_epsilon(N)*3, x_, vals), label = "alpha = $alpha, bias^2 = $(round(bias^2,digits=1)), variance = $(round(var,digits=1)), E(MSE) = $(round(expected_mse,digits=1))", color = plotting_colors[i])
        scatter!(plt, [clamp(μ, true_div-10, true_div+10)], [0.05], color = plotting_colors[i], markersize = 10, label = "mean, alpha = $alpha")
    end
    scatter!(plt, [true_div], [0], label = "true divergence", markersize = 10)
    push!(plots, plt)
end
plot(plots..., layout = (3, 2))

############## distribution of |x| at t = 6 for landau ############
using HCubature, LinearAlgebra, Plots
a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
start_t = 6.5
K(t) = 1 - exp(-(t+start_t)/6)
true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * (2π*K)^(-3/2) * exp(-sum(abs2, x)/(2K))
cdf_norm(r) = hcubature(x -> (norm(x) < r ? 1 : 0)*true_pdf(x, K(0f0)), [-r,-r,-r], [r,r,r], rtol = 0.005, initdiv = 10)[1]
dr = 0.01
r_vals = 0.1:dr:3
cdf_vals = cdf_norm.(r_vals)
pdf_vals = [cdf_vals[i+1] - cdf_vals[i-1] for i in 2:length(cdf_vals)-1] ./ (2dr)
sum(pdf_vals) * dr
plot(r_vals[2:end-1], pdf_vals, title = "pdf of |x| at t = $start_t", xlabel = "|x|", ylabel = "pdf", size = (1600, 1000))

# gaussian soap bubble
using Plots
ds = [1, 3, 10, 100, 10_000]
N = 10^4
plots = []
for d in ds
    data = randn(d, N)
    norms = sum(abs2, data, dims = 1) |> vec .|> sqrt
    plt = histogram(norms, title = "d = $d", xlabel = "|x|", ylabel = "histogram count")
    push!(plots, plt)
end
plot(plots..., layout = (3, 2), size = (1600, 1000))


# allocations in loss
using Flux, BenchmarkTools, TimerOutputs, CUDA
function normsq(z)
    res = zero(eltype(z))
    for zi in z
        res += zi*zi
    end
    res
end

function loss1(s, xs :: AbstractArray{T}, α = T(0.4)) where T
    ζ = randn(T, size(xs))
    denoise_val = ( s(xs .+ α .* ζ) ⋅ ζ - s(xs .- α .* ζ) ⋅ ζ ) / α
    (norm(s(xs))^2 + denoise_val)/num_particles(xs)
end
function loss2(s, xs :: AbstractArray{T}, α = T(0.4)) where T
    @timeit "l1" ζ = randn(T, size(xs))
    @timeit "l2" denoise_val = ( s(xs .+ α .* ζ) ⋅ ζ - s(xs .- α .* ζ) ⋅ ζ ) / α
    @timeit "l3" (normsq(s(xs)) + denoise_val)/num_particles(xs)
end
function loss3(s, xs :: AbstractArray{T}, ζ, α = T(0.4)) where T
    # ζ = CUDA.randn(T, size(xs))
    denoise_val = ( s(xs .+ α .* ζ) ⋅ ζ - s(xs .- α .* ζ) ⋅ ζ ) / α
    (norm(s(xs))^2 + denoise_val)/num_particles(xs)
end

function loss4(s_, xs_ :: AbstractArray{T}, ζ_, α = T(0.4)) where T
    s = gpu(s_); xs = gpu(xs_);
    ζ = gpu(ζ_)
    # ζ = CUDA.randn(T, size(xs))
    denoise_val = ( s(xs .+ α .* ζ) ⋅ ζ - s(xs .- α .* ζ) ⋅ ζ ) / α
    (norm(s(xs))^2 + denoise_val)/num_particles(xs)
end

num_particles(xs :: AbstractArray{T, 2}) where T = size(xs, 2)
xs = randn(Float32, 3, 80_000)
s = Chain(Dense(3 => 100, softsign), Dense(100 => 100, softsign), Dense(100 => 3))
ζ = randn(Float32, size(xs))
ζ_gpu = gpu(ζ)
@btime loss1($s, $xs, 1f-1)
s_gpu = gpu(s); xs_gpu = gpu(xs);
@btime CUDA.@sync loss3($s_gpu, $xs_gpu, $ζ_gpu, 1f-1)
@btime CUDA.@sync loss4($s, $xs, $ζ, 1f-1)
@btime withgradient(s -> loss1(s, $xs, 1f-1), $s)
@btime CUDA.@sync withgradient(s -> loss3(s, $xs_gpu, $ζ_gpu, 1f-1), $s_gpu)
@btime CUDA.@sync withgradient(s -> loss4(s, $xs, ζ, 1f-1), $s)

reset_timer!()
loss2(s, xs, 1f-1)
print_timer()
@btime $s($xs)

# diffeq with gpu
using OrdinaryDiffEq, CUDA, LinearAlgebra
u0 = rand(Float32, 1000)
A = randn(Float32, 1000, 1000)
u0_gpu = gpu(u0)
A_gpu = gpu(A)
f(du, u, p, t) = mul!(du, A, u)
f_gpu(du, u, p, t) = mul!(du, A_gpu, u)
prob = ODEProblem(f, u0, (0.0f0, 1.0f0))
prob_gpu = ODEProblem(f_gpu, u0_gpu, (0.0f0, 1.0f0))
sol = solve(prob, Euler(), dt = 0.01)
sol_gpu = solve(prob_gpu, Euler(), dt = 0.01)
sol_gpu.u[1] - cu(sol.u[1])


# plotting 2d solutions
using Plots
markersize = 5
plot();
scatter!(solution_sbtm[end][1,1,:], solution_sbtm[end][2,1,:], label = "sbtm, t = 10", markersize = markersize);
scatter!(solution_correct[end][1,1,:], solution_correct[end][2,1,:], label = "blob, t = 10", markersize = markersize);
scatter!(xs[1,1,:], xs[2,1,:], label = "initial sample", markersize = markersize);
plot!(title = "pure diffusion", size = (1200, 1200), xlabel = "x", ylabel = "y")

ε = 0.7
correct_pdf(x) = reconstruct_pdf(ε, x, solution_correct[end])
incorrect_pdf(x) = reconstruct_pdf(ε, x, solution_incorrect[end])
sbtm_pdf(x) = reconstruct_pdf(ε, x, solution_sbtm[end])
y = 0
plot(-20:0.01:20, x -> correct_pdf([x,y]), label = "blob");
plot!(-20:0.01:20, x -> sbtm_pdf([x,y]), label = "sbtm");
plot!(-20:0.01:20, x -> pdf(f(ts[end]), [x,y]), label = "true")

# comparing sbtm on cpu and gpu
using BenchmarkTools
no_alloc!(x) = for i in eachindex(x) x[i] = randn() end
alloc(x) = (x = randn(size(x)))
x = randn(3*160_000)
@btime no_alloc!($x) # allocates
@btime alloc($x)

# fpe on gpu
include("src/fpe/sbtm.jl")
d = 2
s = Chain(Dense(d => 100, softsign), Dense(100 => 100, softsign), Dense(100 => d))
n = 100_000
xs, ts, b, D, ρ₀, ρ = pure_diffusion(d, n, 0.01, 0.02)


sg = gpu(s)
xsg = gpu(xs)
tsg = gpu(ts)
CUDA.@time sol_gpu,_,_=sbtm_fpe(xsg, tsg, b, D; s = sg, saveat = tsg[end]) # 12 seconds

@time sol_cpu,_,_=sbtm_fpe(xs, ts, b, D; s = s, saveat = ts[end]) # 276 seconds
maximum(abs, cpu(sol_gpu.u)[end] .- sol_cpu.u[end])


# converting blob to gpu
using LoopVectorization

function blob_score!(score_array :: Array, xs :: Array, pars)
    (ε, diff_norm2s, mol_sum, mols) = pars
    d, n = size(xs)
    mol_sum .= 0
    diff_norm2s .= 0
    score_array .= 0
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        diff_norm2s[p, q] += (xs[k, p] - xs[k, q])^2
    end
    @tturbo for p in 1:n, q in 1:n
        mols[p, q] = exp(-diff_norm2s[p, q]/ε)/sqrt((π*ε)^d)
        mol_sum[p] += mols[p, q]
    end
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        fac = -2. / ε * mols[p, q]
        diff_k = xs[k, p] - xs[k, q]
        score_array[k, p] += fac * diff_k / mol_sum[p]
        score_array[k, p] += fac * diff_k / mol_sum[q]
    end
end

using CUDA
function blob_score_gpu!(score_array, xs, pars)
    (ε, mol_sum) = pars
    d, n = size(xs)
    ker = @cuda launch = false kernel1(mol_sum, xs, ε)
    config = launch_configuration(ker.fun)
    threads = min(length(mol_sum), config.threads)
    blocks = cld(length(mol_sum), threads)
    @cuda threads = threads blocks = blocks kernel1(mol_sum, xs, ε)
    @show threads, blocks

    ker = @cuda launch = false kernel1(mol_sum, xs, ε)
    config = launch_configuration(ker.fun)
    threads = (d, div(min(length(mol_sum), config.threads), d))
    blocks = (1, div(n, threads[2], RoundUp))
    @cuda threads = threads blocks = blocks kernel2(score_array, mol_sum, xs, ε)
    @show threads, blocks

end
function kernel2(score_array, mol_sum, xs, ε)
    d, n = size(xs)
    k = (blockIdx().x-1) * blockDim().x + threadIdx().x
    p = (blockIdx().y-1) * blockDim().y + threadIdx().y
    @inbounds if k <= d && p <= n
        score_array[k, p] = zero(eltype(xs))
        for q in 1:n
            diff_norm2 = zero(eltype(xs))
            for k in 1:d
                diff_norm2 += (xs[k, p] - xs[k, q])^2
            end
            mol_pq = exp(-diff_norm2/ε)/sqrt((π*ε)^d)
            fac = -2 / ε * mol_pq
            diff_k = xs[k, p] - xs[k, q]
            score_array[k, p] += fac * (diff_k) / mol_sum[p]
            score_array[k, p] += fac * (diff_k) / mol_sum[q]
        end
    end
end

function kernel1(mol_sum, xs, ε)
    d, n = size(xs)
    p = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if p <= n
        accum = zero(eltype(xs))
        for q in 1:n
            norm_diff = zero(eltype(xs))
            for k in 1:d
                norm_diff += (xs[k, p] - xs[k, q])^2
            end
            mol_pq = exp(-norm_diff/ε)/sqrt((π*ε)^d)
            accum += mol_pq
        end
        mol_sum[p] = accum
    end
    return
end

CUDA.allowscalar(false)
n = 40_000
d = 10
xs = CUDA.randn(d, n)
mol_sum = CUDA.zeros(n)
score_array = CUDA.zeros(d, n)
ε = 1f-1

CUDA.@time blob_score_gpu!(score_array, xs, (ε, mol_sum))

ker = @cuda launch = false kernel1(mol_sum, xs, ε)
config = launch_configuration(ker.fun)
threads = min(length(mol_sum), config.threads)
blocks = cld(length(mol_sum), threads)

ker2 = @cuda launch = false kernel2(score_array, mol_sum, xs, ε)
config = launch_configuration(ker2.fun)
threads = min(length(mol_sum), config.threads)
blocks = cld(size(score_array,1)*size(score_array,2), threads)




using Flux
score_array_cpu = zeros(Float32, d, n)
x_cpu = cpu(xs)
diff_norm2s = zeros(Float32, n, n)
mol_sum_cpu = zeros(Float32, n)
term1 = zeros(Float32, d, n)
term2 = zeros(Float32, d, n)
mols = zeros(Float32, n, n)

# @time blob_score!(score_array_cpu, x_cpu, (ε, diff_norm2s, mol_sum_cpu, term1, term2, mols))
@time blob_score!(score_array_cpu, x_cpu, (ε, diff_norm2s, mol_sum_cpu, mols))

maximum(abs, cpu(mol_sum) .- mol_sum_cpu)
maximum(abs, cpu(score_array) .- score_array_cpu)

xs = cpu(xs)
diff_norm2s = zeros(n, n)
for p in 1:n, q in 1:n, k in 1:d
    diff_norm2s[p, q] += (xs[k, p] - xs[k, q])^2
end
maximum(abs, cpu(diff_norms) .- diff_norm2s)


ker = @cuda launch=false kernel1(diff_norms, xs)
config = launch_configuration(ker.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

diff_norms = CUDA.ones(n, n)
@cuda threads = (2,2) blocks = 6 kernel1(diff_norms, xs)

using CUDA
a = CUDA.zeros(100_000)

function kernel(a)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(a)
        a[i] += 1
    end
    return
end

@cuda threads=1024 blocks = (length(a) ÷ 1024 + 1) kernel(a)

# benchmarking cpu vs gpu fpe
using Random, CUDA
include("src/fpe/sbtm.jl")
include("src/fpe/blob.jl")
CUDA.allowscalar(false)
d = 3
n = 10^3
ε = epsilon(d, n)
Random.seed!(1234)
xs, ts, b, D, ρ₀, ρ = pure_diffusion(d, n, 0.05)
@show n

# blob
@time blob_fpe(xs, ts, b, D; ρ₀ = ρ₀, ε = ε, saveat = ts[end])
CUDA.@time blob_fpe(xs, ts, b, D; ρ₀ = ρ₀, ε = ε, saveat = ts[end], usegpu = true)

# sbtm
@time sbtm_fpe(xs, ts, b, D; ρ₀ = ρ₀, ε = ε, saveat = ts[end], verbose = 1)
CUDA.@time sol, models, _ = sbtm_fpe(xs, ts, b, D; ρ₀ = ρ₀, ε = ε, saveat = ts[[1,2,end]], usegpu = true, verbose = 1, record_models = true)

cu(sol)

# tsg = cu(Float32.(0:0.01:1))
# dt = 1f-2
# affect!(integrator) = nothing
# f!(du, u, p, t) = (du .= u)
# ode_problem = ODEProblem(f!, cu([1.]), (0f0, 1f0), nothing)

# cbg = PresetTimeCallback(tsg, affect!, save_positions=(false,false))
# cb = PresetTimeCallback(0:dt:1, affect!, save_positions=(false,false))
# solution = solve(ode_problem, alg = Euler(), dt = dt, callback = cbg) # ERROR: CuArray only supports element types that are allocated inline.
# solution = solve(ode_problem, alg = Euler(), dt = dt, callback = cb) # ERROR: CuArray only supports element types that are allocated inline.
# solution = solve(ode_problem, alg = Euler(), dt = dt) # Success

using OrdinaryDiffEq, CUDA, LinearAlgebra
u0 = cu(rand(1000))
A = cu(randn(1000, 1000))
f(du, u, p, t) = mul!(du, A, u)
prob = ODEProblem(f, u0, (0.0f0, 1.0f0)) # Float32 is better on GPUs!
sol_ = solve(prob, Tsit5())
sol_.u = cpu(sol_.u)
cpu(sol_) # returns a CuArray instead of a solution with 

# pre train diffusion model
using JLD2
n = 20_000
for d in [2,3,5,10]
    xs, ts, b, D, ρ₀, ρ = pure_diffusion(d, n)
    s = Chain(Dense(d => 100, softsign), Dense(100 => d))
    initialize_s!(s, ρ₀, xs, loss_tolerance = 1e-4, verbose = 2, max_iter = 10^5)
    save("models/diffusion_model_d_$(d)_n_$(n)_start_1.jld2", "s", s)
end