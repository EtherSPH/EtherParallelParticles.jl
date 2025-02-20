#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 20:18:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Dimension" begin
    @test EtherParallelParticles.Environment.dimension(EtherParallelParticles.Environment.Dimension2D) == 2
    @test EtherParallelParticles.Environment.dimension(EtherParallelParticles.Environment.Dimension3D) == 3
end
