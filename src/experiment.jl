using JLD

include("utils.jl")
include("sbtm.jl")
include("jhu.jl")

# first example from paper
xs, Δts, b, D, ρ₀, target = moving_trap()

ε = 1.
t1 = time()
jhu_trajectories = jhu(xs, Δts, b, D, ε)
t2 = time()
# animation = animate_2d(trajectories, "jhu", Δts; target = target, plot_every = 5, fps = 5)

println("Done with jhu. Took $(t2-t1) seconds.")

s = initialize_s(ρ₀, xs, 32, 1)
epochs = 10
t3 = time()
println("Done with NN initialization. Took $(t3-t2) seconds.")
sbtm_trajectories, losses = sbtm(xs, Δts, b, D, s; optimiser = Adam(10^-3), epochs = epochs, record_losses = true, verbose = false)
t4 = time()
# animation = animate_2d(trajectories, "sbtm", Δts, samples = 2, fps = 10, target = target)
# loss_plot = plot_losses(losses, epochs)

println("Done with sbtm. Took $(t4-t3) seconds.")

save("moving_trap_data.jld", "sbtm_trajectories", sbtm_trajectories, "losses", losses, "jhu_trajectories", jhu_trajectories, "timestamps", [t1, t2, t3, t4])

print("Done with saving")