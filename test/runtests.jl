using Test, TimerOutputs

@time begin
    @timeit "fokker-planck tests" @testset "fokker-planck tests" begin include("fpe.jl") end
    @timeit "landau tests"        @testset "landau tests" begin include("landau.jl") end
end