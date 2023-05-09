using Distributions, TimerOutputs, JLD2, Flux, Plots
using Random: seed!

include("blob.jl")
include("sbtm.jl")
include("../utils.jl")

start_time = 6
pre_trained_s = load("models/landau_model_n_4000_start_6.jld2", "s")
a(K) = (5K-3)/(2K)
b(K) = (1-K)/(2K^2)
K(t) = 1 - exp(-(t+start_time)/6)
true_pdf(x, K) = (a(K) + b(K)*sum(abs2, x)) * (2π*K)^(-3/2) * exp(-sum(abs2, x)/(2K))

ns = [4000, 8000, 16_000]
lrs = [(1e-4)/4, 1e-4, 4*1e-4]
epoch_nums = [10, 25, 50, 100]
alphas = [0.01, 0.1, 0.4]
# ns = [500]
# lrs = [(1e-4)/4]
# epoch_nums = [10]
# alphas = [0.01]

# generate data
reset_timer!()
@timeit "generate data" for (i, n) in enumerate(ns)
    seed!(n)
    xs, ts, ρ = landau(n, start_time)
    ρ₀(x) = ρ(x,K(0))
    saveat = ts[[1, end]]
    s_ = deepcopy(pre_trained_s)
    @timeit "n = $n NN init" initialize_s!(s_, ρ₀, xs, loss_tolerance = 1e-4, verbose = 0, max_iter = 10^4)
    @timeit "n = $n solving" for (j, lr) in enumerate(lrs), (k, epochs) in enumerate(epoch_nums), (p, alpha) in enumerate(alphas)
        println("i,j,k,p = $((i,j,k,p))")
        println("    n = $n, lr = $lr, epochs = $epochs, alpha = $alpha")
        solution, models, _ = sbtm_landau(xs, ts; s = s_, verbose = 0, saveat = saveat, record_models = true, denoising_alpha = alpha, epochs = epochs, optimiser = Adam(lr))
        save("hyperparameter_sweep/landau_n_$(n)_lr_$(lr)_epochs_$(epochs)_alpha_$(alpha).jld2", "solution", solution, "models", models, "saveat", saveat)
    end
end

# compute errors
model_errors = ones(length(ns), length(lrs), length(epoch_nums), length(alphas))
pdf_errors = ones(length(ns), length(lrs), length(epoch_nums), length(alphas))
@timeit "L2 error computation" for (i, n) in enumerate(ns), (j, lr) in enumerate(lrs), (k, epochs) in enumerate(epoch_nums), (p, alpha) in enumerate(alphas)
    _, ts, _ = landau(n, start_time)
    data = load("hyperparameter_sweep/landau_n_$(n)_lr_$(lr)_epochs_$(epochs)_alpha_$(alpha).jld2")
    u = data["solution"][end]
    s = data["models"][end]
    model_l2_error = (sum(abs2, s(u) - score(z -> true_pdf(z, K(ts[end])), u)) / n)^(1/2)
    pdf_l2_error = Lp_error_slice(u, x->true_pdf(x, K(ts[end])); k=2, p=2, verbose = 0, xlim = 2.5, rtol=0.02)
    model_errors[i,j,k,p] = model_l2_error
    pdf_errors[i,j,k,p] = pdf_l2_error
end
print_timer()

save("hyperparameter_sweep/landau_error_matrix.jld2", "model_errors", model_errors, "pdf_errors", pdf_errors, "ns", ns, "lrs", lrs, "epoch_nums", epoch_nums, "alphas", alphas, "timer", TimerOutputs.get_defaulttimer())

# sort the matrices to find the best hyperparameters
data = load("hyperparameter_sweep/landau_error_matrix.jld2")
model_errors = data["model_errors"]
pdf_errors = data["pdf_errors"]
ns = data["ns"]
lrs = data["lrs"]
epoch_nums = data["epoch_nums"]
alphas = data["alphas"]

for (i,n) in enumerate(ns)
    @show n
    zipped = zip(CartesianIndices(model_errors[i,:,:,:]), Iterators.product(lrs, epoch_nums, alphas), model_errors[i,:,:,:]) |> collect |> vec
    sort!(zipped, by = x -> x[3])
    cis, (lr, epochs, alpha), error = first(zipped)
    println("    lr = $lr, epochs = $epochs, alpha = $alpha, model l2 error = $error, cis = $cis")
end

for (i,n) in enumerate(ns)
    @show n
    zipped = zip(CartesianIndices(pdf_errors[i,:,:,:]), Iterators.product(lrs, epoch_nums, alphas), pdf_errors[i,:,:,:]) |> collect |> vec
    sort!(zipped, by = x -> x[3])
    cis, (lr, epochs, alpha), error = first(zipped)
    println("    lr = $lr, epochs = $epochs, alpha = $alpha, model l2 error = $error, cis = $cis")
end

model_plt = plot(title="model comparison");
plot!(model_plt,-2:0.01:2, x -> score(z -> true_pdf(z, K(0.5)), [x,0,0])[1], label = "true score at t = 6.5");
pdf_plt = plot(title = "pdf comparison");
plot!(pdf_plt, -2:0.01:2, x -> true_pdf([x,0,0], K(0.5)), label = "true pdf at t = 6.5");

n = 16_000; lr = lrs[1]; epochs = epoch_nums[2]; alpha = alphas[2];
data = load("hyperparameter_sweep/landau_n_$(n)_lr_$(lr)_epochs_$(epochs)_alpha_$(alpha).jld2")
plot!(model_plt,-2:0.01:2, x -> data["s"]([x,0,0])[1], label = "n = $n, lr = $lr, epochs = $epochs, alpha = $alpha, error = $(model_errors[end, 1, 2, 2])");
plot!(pdf_plt, -2:0.01:2, x -> reconstruct_pdf([x,0,0], data["u"]), label = "n = $n, lr = $lr, epochs = $epochs, alpha = $alpha, error = $(pdf_errors[end, 1, 2, 2])");

n = 16_000; lr = lrs[3]; epochs = epoch_nums[4]; alpha = alphas[2];
data = load("hyperparameter_sweep/landau_n_$(n)_lr_$(lr)_epochs_$(epochs)_alpha_$(alpha).jld2")
plot!(model_plt,-2:0.01:2, x -> data["s"]([x,0,0])[1], label = "n = $n, lr = $lr, epochs = $epochs, alpha = $alpha, error = $(model_errors[end, end, end, 2])");
plot!(pdf_plt, -2:0.01:2, x -> reconstruct_pdf([x,0,0], data["u"]), label = "n = $n, lr = $lr, epochs = $epochs, alpha = $alpha, error = $(pdf_errors[end, end, end, 2])");

plot(model_plt, pdf_plt, layout = (1,2), size = (1600, 1200))