using LoopVectorization, CUDA, Flux

function score_params(xs :: Array, ε :: T) where T
    n = num_particles(xs)
    d = get_d(xs)
    diff_norm2s = zeros(T, n, n)
    mol_sum = zeros(T, n)
    mols = zeros(T, n, n)
    (ε, diff_norm2s, mol_sum, mols)
end

function score_params(xs :: CuArray, ε :: T) where T
    n = num_particles(xs)
    d = get_d(xs)
    mol_sum = zeros(T, n) |> gpu
    (ε, mol_sum)
end

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

function blob_score!(score_array :: CuArray, xs :: CuArray, pars)
    (ε, mol_sum) = pars
    d, n = size(xs)
    ker = @cuda launch = false kernel1(mol_sum, xs, ε)
    config = launch_configuration(ker.fun)
    threads = min(length(mol_sum), config.threads)
    blocks = cld(length(mol_sum), threads)
    @cuda threads = threads blocks = blocks kernel1(mol_sum, xs, ε)

    ker = @cuda launch = false kernel1(mol_sum, xs, ε)
    config = launch_configuration(ker.fun)
    threads = (d, div(min(length(mol_sum), config.threads), d))
    blocks = (1, div(n, threads[2], RoundUp))
    @cuda threads = threads blocks = blocks kernel2(score_array, mol_sum, xs, ε)
end

function kernel2(score_array, mol_sum, xs, ε)
    d, n = size(xs)
    k = (blockIdx().x-1) * blockDim().x + threadIdx().x
    p = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if k <= d && p <= n
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
    if p <= n
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