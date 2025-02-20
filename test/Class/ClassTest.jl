#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/14 15:57:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Class" begin
    include("DomainTest.jl")
    include("ParticleSystem/ParticleSystemTest.jl")
    include("NeighbourSystem/NeighbourSystemTest.jl")
end
