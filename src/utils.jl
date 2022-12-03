function moving_trap(N = 5, num_samples = 2, num_timestamps = 20)
    d = 2 # dimension of each particle
    a = Float32(2.) # trap amplitude
    w = Float32(1.) # trap frequency
    α = Float32(.5) # repelling force
    Δts = 0.01*ones(Float32, num_timestamps) # time increments
      
    # define drift vector field b and diffusion matrix D
    β(t) = a*Float32[cos(π*w*t), sin(π*w*t)]
    function b(x, t)
        attract = β(t) .- x
        repel = α * (x .- mean(x, dims = 2))
        attract + repel
    end
    D(x, t) = Float32(0.)
    
    # draw samples
    ρ₀ = MvNormal(β(0.), 0.25*I(d))
    xs = convert(Array{Float32, 3}, reshape(rand(ρ₀, N*num_samples), d, N, num_samples))

    # positions of moving trap
    target = hcat(β.(vcat(0., cumsum(Δts)))...)

    xs, Δts, b, D, ρ₀, target
end

function attractive_origin()
    d = 2
    N = 1
    num_samples = 9
    Δts = 0.01*ones(Float32, 5)
    b(x, t) = -x
    D(x, t) = Float32(0.1)
    xs = reshape(Float32[-1  0  1 -1  0  1 -1  0  1;
    -1 -1 -1  0  0  0  1  1  1], d, N, num_samples);
    s = Chain(
      Dense(d => 50, relu),
      Dense(50 => d))
    print("Initializing s...")
    initialize_s!(s, MvNormal(zeros(d), I(d)), xs, ε = 10^-2)
    xs, Δts, b, D, s
end

function animate_2d(trajectories, title, Δts; samples = 1, fps = 5, target = zeros(2,length(Δts)+1), plot_every = 1)
    xmax = maximum([maximum(target[1,:]), maximum(abs.(trajectories[1,:,samples,:]))]) + 0.5
    ymax = maximum([maximum(target[2,:]), maximum(abs.(trajectories[2,:,samples,:]))]) + 0.5
    @show xmax, ymax
    ts = round.(vcat(zero(Δts[1]), cumsum(Δts)), digits = 4)
    plot_size = (700,700)
    p = scatter(trajectories[1,:,samples,1], trajectories[2,:,samples,1], 
                title = "$title, time $(ts[1])",
                label = nothing, 
                color=RGB(0.0, 0.0, 1.0), 
                xlims = (-xmax, xmax), ylims = (-ymax, ymax),
                markersize = 3, size = plot_size);
    target !== nothing && scatter!(p, [target[1, 1]], [target[2, 1]], markershape = :star, label = "target", color = :red)
    
    anim = @animate for k ∈ axes(trajectories, 4)[1:plot_every:end]
        λ = k/size(trajectories, 4)
        red = λ > 0.5 ? 2. *(λ - 0.5) : 0.
        green = 1. - abs(1. - 2. * λ)
        blue = λ < 0.5 ? 2. * (0.5-λ) : 0.
        target !== nothing && scatter!(p, [target[1, k]], [target[2, k]], markershape = :star, color = :red, label = nothing)
        scatter!(p, vec(trajectories[1,:,samples,k]), vec(trajectories[2,:,samples,k]), 
                  title = "$title, time $(ts[k])",
                  label = nothing, 
                  color = RGB(red, green, blue), 
                  xlims = (-xmax, xmax), ylims = (-ymax, ymax),
                  markersize = 3, size = plot_size)
  end
  gif(anim, "$(title)_anim_fps$fps.gif", fps = fps)
end