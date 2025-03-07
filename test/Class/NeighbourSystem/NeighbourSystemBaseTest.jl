#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/17 19:39:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "NeighbourSystemBase $DEVICE" begin
    domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
    # 5 * 4 = 20 cells
    # 4 | 6 | 6 | 6 | 4
    # --|---|---|---|--
    # 6 | 9 | 9 | 9 | 6
    # --|---|---|---|--
    # 6 | 9 | 9 | 9 | 6
    # --|---|---|---|--
    # 4 | 6 | 6 | 6 | 4
    neighbour_system_base = EtherParallelParticles.Class.NeighbourSystemBase(parallel, domain)
    @test EtherParallelParticles.Environment.toHost(parallel, neighbour_system_base.neighbour_cell_index_count_) ==
          [4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
    @test size(neighbour_system_base.neighbour_cell_index_count_) == (20,)
    @test size(neighbour_system_base.neighbour_cell_index_list_) == (20, 9)
end
