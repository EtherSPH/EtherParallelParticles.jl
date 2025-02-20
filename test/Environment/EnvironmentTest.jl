#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 20:18:02
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Environment" begin
    @testset "Dimension" begin
        include("DimensionTest.jl")
    end
    @testset "Parallel" begin
        include("ParallelTest.jl")
    end
end
