#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/17 19:03:30
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "NeighbourSystem" begin
    include("NeighbourSystemBaseTest.jl")
    include("ActivePairTest.jl")
    include("PeriodicBoundaryTest.jl")
    if USING_CPU == true
        @testset "NeighbourSystem CPU" begin
            include("../../cpu_test_head.jl")
            domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
            active_pair = [1 => 1, 1 => 2, 2 => 1]
            periodic_boundary_policy = EtherParallelParticles.Class.NonePeriodicBoundaryPolicy
            neighbour_system =
                EtherParallelParticles.Class.NeighbourSystem(parallel, domain, active_pair, periodic_boundary_policy)
            EtherParallelParticles.Class.clean!(neighbour_system)
            # * base
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.base_.contained_particle_index_count_,
            ) == zeros(IT, EtherParallelParticles.Class.get_n(domain))
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.base_.neighbour_cell_index_count_,
            ) == IT[4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
            @test size(neighbour_system.base_.neighbour_cell_index_count_) == (20,)
            @test size(neighbour_system.base_.neighbour_cell_index_list_) == (20, 9)
            # * active pair
            @test neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.active_pair_.adjacency_matrix_,
            ) == IT[
                1 1
                1 0
            ]
            @test size(neighbour_system.periodic_boundary_.neighbour_cell_relative_position_list_) == (1, 1, 1)
        end
    end
    if USING_CUDA == true
        @testset "NeighbourSystem CUDA" begin
            include("../../cuda_test_head.jl")
            domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
            active_pair = [1 => 1, 1 => 2, 2 => 1]
            periodic_boundary_policy = EtherParallelParticles.Class.NonePeriodicBoundaryPolicy
            neighbour_system =
                EtherParallelParticles.Class.NeighbourSystem(parallel, domain, active_pair, periodic_boundary_policy)
            EtherParallelParticles.Class.clean!(neighbour_system)
            # * base
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.base_.contained_particle_index_count_,
            ) == zeros(IT, EtherParallelParticles.Class.get_n(domain))
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.base_.neighbour_cell_index_count_,
            ) == IT[4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
            @test size(neighbour_system.base_.neighbour_cell_index_count_) == (20,)
            @test size(neighbour_system.base_.neighbour_cell_index_list_) == (20, 9)
            # * active pair
            @test neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.active_pair_.adjacency_matrix_,
            ) == IT[
                1 1
                1 0
            ]
            @test size(neighbour_system.periodic_boundary_.neighbour_cell_relative_position_list_) == (1, 1, 1)
        end
    end
    if USING_ONEAPI == true
        @testset "NeighbourSystem ONEAPI" begin
            include("../../oneapi_test_head.jl")
            domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
            active_pair = [1 => 1, 1 => 2, 2 => 1]
            periodic_boundary_policy = EtherParallelParticles.Class.NonePeriodicBoundaryPolicy
            neighbour_system =
                EtherParallelParticles.Class.NeighbourSystem(parallel, domain, active_pair, periodic_boundary_policy)
            EtherParallelParticles.Class.clean!(neighbour_system)
            # * base
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.base_.contained_particle_index_count_,
            ) == zeros(IT, EtherParallelParticles.Class.get_n(domain))
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.base_.neighbour_cell_index_count_,
            ) == IT[4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
            @test size(neighbour_system.base_.neighbour_cell_index_count_) == (20,)
            @test size(neighbour_system.base_.neighbour_cell_index_list_) == (20, 9)
            # * active pair
            @test neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
            @test EtherParallelParticles.Environment.toHost(
                parallel,
                neighbour_system.active_pair_.adjacency_matrix_,
            ) == IT[
                1 1
                1 0
            ]
            @test size(neighbour_system.periodic_boundary_.neighbour_cell_relative_position_list_) == (1, 1, 1)
        end
    end
end
