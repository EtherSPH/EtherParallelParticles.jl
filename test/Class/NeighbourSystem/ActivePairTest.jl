#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/17 19:03:11
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "ActivePair" begin
    if USING_CPU == true
        @testset "ActivePair CPU" begin
            include("../../Head/cpu_test_head.jl")
            pair_vector = [1 => 1, 1 => 2, 2 => 1, 1 => 3, 3 => 1]
            active_pair = EtherParallelParticles.Class.ActivePair(parallel, pair_vector)
            @test active_pair.pair_vector_ ==
                  [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1), IT(1) => IT(3), IT(3) => IT(1)]
            @test EtherParallelParticles.Environment.toHost(parallel, active_pair.adjacency_matrix_) == IT[
                1 1 1
                1 0 0
                1 0 0
            ]
        end
    end
    if USING_CUDA == true
        @testset "ActivePair CUDA" begin
            include("../../Head/cuda_test_head.jl")
            pair_vector = [1 => 1, 1 => 2, 2 => 1, 1 => 3, 3 => 1]
            active_pair = EtherParallelParticles.Class.ActivePair(parallel, pair_vector)
            @test active_pair.pair_vector_ ==
                  [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1), IT(1) => IT(3), IT(3) => IT(1)]
            @test EtherParallelParticles.Environment.toHost(parallel, active_pair.adjacency_matrix_) == IT[
                1 1 1
                1 0 0
                1 0 0
            ]
        end
    end
    if USING_ONEAPI == true
        @testset "ActivePair ONEAPI" begin
            include("../../Head/oneapi_test_head.jl")
            pair_vector = [1 => 1, 1 => 2, 2 => 1, 1 => 3, 3 => 1]
            active_pair = EtherParallelParticles.Class.ActivePair(parallel, pair_vector)
            @test active_pair.pair_vector_ ==
                  [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1), IT(1) => IT(3), IT(3) => IT(1)]
            @test EtherParallelParticles.Environment.toHost(parallel, active_pair.adjacency_matrix_) == IT[
                1 1 1
                1 0 0
                1 0 0
            ]
        end
    end
end
