# learning Flux
using Flux

maybewrap(x) = x
maybewrap(x::T) where {T <: Number} = T[x]

function plot_s_1d(s, xs)
    x_coordinates = minimum(xs) : (maximum(xs) - minimum(xs))/99 : maximum(xs)
    s_values = vcat(s.(eachcol(reshape(x_coordinates, 1, 100)))...)
    p = plot(x_coordinates, s_values)
    scatter!(p, reshape(xs, size(xs, 2)), vcat(s.(eachcol(xs))...))
end

function custom_train!(s, xs; optimiser = Adam(10^-4), num_steps = 25)
    θ = params(s)
    grads = gradient(() -> loss(s, xs), θ)
    for _ in 1:num_steps
      Flux.update!(optimiser, θ, grads)
    end
end

# learning the sin function
function learn_sin()
    s = Chain(
      maybewrap,
      Dense(1 => 50, relu),
      Dense(50 => 50, relu),
      Dense(50 => 50, relu),
      Dense(50 => 50, relu),
      Dense(50 => 1))
    f(x) = sin.(6.28 .* x .+ 3.14)
    x_train = hcat(0.5 : 3/100 : 3.5 ...)
    data = [(x_train, f.(x_train))]
    mse_loss(x, y) = Flux.Losses.mse(s(x), y)
    p = plot(x_train[1,:], vcat(s.(x_train)[1,:]...));
    for i in 1:5*10^3
        train!(mse_loss, params(s), data, Descent(10^-1))
        if i%10^3 == 0
            @show mse_loss(data[1]...)
            plot!(p, x_train[1,:], vcat(s.(x_train)[1,:]...))
        end
    end
    s, p
end

# experimenting with (over)fitting on 3 points
s, p = learn_sin()
xs = [1. 2. 3.;] 
loss(s, xs)
custom_train!(s, xs, optimiser = Adam(10^-3), num_steps = 100)
loss(s, xs)
plot_s_1d(s,xs)

# Findings: 
# -- relu works much better than swish (faster convergence)
# -- with random initialization the NN does not converge to the sawtooth shape I’d expect
# -- from sin-wave initialization, the function stays in the approximate sawtooth shape but only for low learning rate (~10^-3)



# approximating x -> -x
using Flux, Random
Random.seed!(123)
actual(x) = -x
x_train = hcat(0:5...)
y_train = actual.(x_train)
for k in 1:6
    predict = Chain(
        Dense(d => d))
    parameters = Flux.params(predict)
    loss_(x, y) = sum( (predict(x) - y).^2 ) / sum(y_train.^2) ;
    learning_rate = 10. ^-k
    opt = Descent(learning_rate)
    loss_(x_train, y_train) # 1.241
    for _ in 1:10000 
        grads = gradient(() -> loss_(x_train, y_train), parameters)
        Flux.Optimise.update!(opt, Flux.params(predict), grads) 
    end
    @show learning_rate, loss_(x_train, y_train) 
    # (learning_rate, loss_(x_train, y_train)) = (0.1, 0.18440121f0)
    # (learning_rate, loss_(x_train, y_train)) = (0.010000000000000002, 0.17265555f0)
    # (learning_rate, loss_(x_train, y_train)) = (0.001, 0.4768501f0)
    # (learning_rate, loss_(x_train, y_train)) = (0.0001, 5.737957f0)
    # (learning_rate, loss_(x_train, y_train)) = (1.0e-5, 0.47478837f0)
    # (learning_rate, loss_(x_train, y_train)) = (1.0e-6, 0.012801164f0)
    # NOTE: loss is never close to zero!
end

# p=plot(xs[1,1,:], ys[1,1,:]);
# plot!(p, xs[1,1,:], s(xs)[1,1,:])


# experimenting with train! vs update!
using Flux: params, train!
s = Chain(
  maybewrap,
  Dense(1 => 50, relu),
  Dense(50 => 50, relu),
  Dense(50 => 50, relu),
  Dense(50 => 50, relu),
  Dense(50 => 1))
s_ = deepcopy(s)
f(x) = sin.(6.28 .* x .+ 3.14)
x_train = hcat(0.5 : 3/100 : 3.5 ...)
data = [(x_train, f.(x_train))]

mse_loss(x, y) = Flux.Losses.mse(s(x), y)
train!(mse_loss, params(s), data, Descent())
    
grads = gradient(() -> sum( Flux.Losses.mse(s_(x), y) for (x,y) in zip(data[1]...))/size(x_train, 2), params(s_))
Flux.update!(Descent(), params(s_), grads)

s(1.0)
s_(1.0)