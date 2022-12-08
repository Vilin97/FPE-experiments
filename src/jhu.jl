using Zygote

include("utils.jl")

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
        xs_k = @view trajectories[:, :, :, k]
        for (j, x) in enumerate(eachslice(xs_k, dims = 3))
            trajectories[:, :, j, k+1] = propagate(ε, x, xs_k, t, Δt, b, D)
        end
        t += Δt
    end
end
