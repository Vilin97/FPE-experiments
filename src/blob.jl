using LoopVectorization

function blob_score!(score_array, xs, pars)
    (ε, diff_norm2s, mol_sum, term1, term2, mols) = pars
    d, n = size(xs)
    mol_sum .= zero(ε)
    diff_norm2s .= zero(ε)
    term1 .= zero(ε)
    term2 .= zero(ε)
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
        term1[k, p] += fac * diff_k / mol_sum[p]
        term2[k, p] += fac * diff_k / mol_sum[q]
    end
    score_array .= term1 .+ term2
end