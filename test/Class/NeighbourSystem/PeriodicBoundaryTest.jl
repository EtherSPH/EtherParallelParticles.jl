#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/17 19:53:29
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "PeriodicBoundary" begin
    if USING_CPU == true
        @testset "PeriodicBoundary CPU" begin
            include("../../Head/cpu_test_head.jl")
            domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
            # 5 * 4 = 20 cells
            # 4 | 6 | 6 | 6 | 4
            # --|---|---|---|--
            # 6 | 9 | 9 | 9 | 6
            # --|---|---|---|--
            # 6 | 9 | 9 | 9 | 6
            # --|---|---|---|--
            # 4 | 6 | 6 | 6 | 4
            periodic_boundary = EtherParallelParticles.Class.PeriodicBoundary(
                parallel,
                domain,
                EtherParallelParticles.Class.NonePeriodicBoundaryPolicy,
            )
            @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (1, 1, 1)
            periodic_boundary = EtherParallelParticles.Class.PeriodicBoundary(
                parallel,
                domain,
                EtherParallelParticles.Class.PeriodicBoundaryPolicy2DAlongX,
            )
            @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (20, 9, 2)
        end
    end
    if USING_CUDA == true
        @testset "PeriodicBoundary CUDA" begin
            include("../../Head/cuda_test_head.jl")
            domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
            # 5 * 4 = 20 cells
            # 4 | 6 | 6 | 6 | 4
            # --|---|---|---|--
            # 6 | 9 | 9 | 9 | 6
            # --|---|---|---|--
            # 6 | 9 | 9 | 9 | 6
            # --|---|---|---|--
            # 4 | 6 | 6 | 6 | 4
            periodic_boundary = EtherParallelParticles.Class.PeriodicBoundary(
                parallel,
                domain,
                EtherParallelParticles.Class.NonePeriodicBoundaryPolicy,
            )
            @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (1, 1, 1)
            periodic_boundary = EtherParallelParticles.Class.PeriodicBoundary(
                parallel,
                domain,
                EtherParallelParticles.Class.PeriodicBoundaryPolicy2DAlongX,
            )
            @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (20, 9, 2)
        end
    end
    if USING_ONEAPI == true
        @testset "PeriodicBoundary ONEAPI" begin
            include("../../Head/oneapi_test_head.jl")
            domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
            # 5 * 4 = 20 cells
            # 4 | 6 | 6 | 6 | 4
            # --|---|---|---|--
            # 6 | 9 | 9 | 9 | 6
            # --|---|---|---|--
            # 6 | 9 | 9 | 9 | 6
            # --|---|---|---|--
            # 4 | 6 | 6 | 6 | 4
            periodic_boundary = EtherParallelParticles.Class.PeriodicBoundary(
                parallel,
                domain,
                EtherParallelParticles.Class.NonePeriodicBoundaryPolicy,
            )
            @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (1, 1, 1)
            periodic_boundary = EtherParallelParticles.Class.PeriodicBoundary(
                parallel,
                domain,
                EtherParallelParticles.Class.PeriodicBoundaryPolicy2DAlongX,
            )
            @test size(periodic_boundary.neighbour_cell_relative_position_list_) == (20, 9, 2)
        end
    end
end
