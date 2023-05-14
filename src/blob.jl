using LoopVectorization, CUDA

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
    mol_sum = CUDA.zeros(T, n)
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
    threads = 512
    CUDA.@sync @cuda threads = threads blocks = (n ÷ threads + 1) kernel1(mol_sum, xs, ε)
    threads = (d, 384 ÷ d)
    @cuda threads = threads blocks = (1, n ÷ threads[2] + 1) kernel2(score_array, mol_sum, xs, ε)
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
        mol_sum[p] = zero(eltype(xs))
        for q in 1:n
            norm_diff = zero(eltype(xs))
            for k in 1:d
                norm_diff += (xs[k, p] - xs[k, q])^2
            end
            mol_pq = exp(-norm_diff/ε)/sqrt((π*ε)^d)
            mol_sum[p] += mol_pq
        end
    end
    return
end