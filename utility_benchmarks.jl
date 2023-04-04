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
@btime normsq($z) # 3.600 ns
@btime normsq_lv($z) # 7.800 ns

z = rand(100)
@btime norm($z)^2 # 43.794 ns
@btime ($z)' * $z # 30.050 ns
@btime sum(x -> x^2, $z) # 14.314 ns
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
landau_f_test2!(dxs, xs, p, 0.)
print_timer()

reset_timer!()
landau_f_test3!(dxs, xs, p, 0.)
print_timer()
@btime landau_f_test2!($dxs, $xs, $p, 0.)
@btime landau_f_test3!($dxs, $xs, $p, 0.)

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