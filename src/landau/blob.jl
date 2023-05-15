using DifferentialEquations, LoopVectorization, TimerOutputs
include("../utils.jl")
include("../blob.jl")

function blob_landau(xs, ts; ε=0.025, kwargs...)
    T = typeof(ε)
    solution = blob_landau_solve(T.(xs), T.(ts), ε; kwargs...)
end

function blob_landau_solve(xs, ts :: AbstractVector{T}, ε :: T; saveat, verbose = 0, kwargs...) where T
    verbose > 0 && println("Blob method. n = $(num_particles(xs)), ε = $ε.")
    tspan = (ts[1], ts[end])
    score_params_ = score_params(xs, ε)
    score_values_temp = zero(xs)
    pars = (score_values_temp, score_params_)
    
    ode_problem = ODEProblem(landau_f_blob!, xs, tspan, pars)
    solution = solve(ode_problem, saveat = saveat, alg = Euler(), tstops = ts)
end

function landau_f_blob!(dxs, xs, pars, t)
    score_values, score_params = pars
    n = num_particles(xs)
    @timeit "compute score" blob_score!(score_values, xs, score_params)
    @timeit "propagate particles" @turbo for p = 1:n
        Base.Cartesian.@nexprs 3 i -> dx_i = zero(eltype(dxs))
        for q = 1:n
            dotzv = zero(eltype(dxs))
            normsqz = zero(eltype(dxs))
            Base.Cartesian.@nexprs 3 i -> begin
                z_i = xs[i, p] - xs[i, q]
                v_i = score_values[i, q] - score_values[i, p]
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