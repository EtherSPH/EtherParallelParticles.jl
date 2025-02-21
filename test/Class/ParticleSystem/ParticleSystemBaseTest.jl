#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/18 15:36:33
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "ParticleSystemBase" begin
    if USING_CPU == true
        @testset "ParticleSystemBase CPU" begin
            include("../../Head/cpu_test_head.jl")
            n_capacity = 100
            int_n_capacity = 20
            float_n_capacity = 30
            particle_system_base =
                EtherParallelParticles.Class.ParticleSystemBase(parallel, n_capacity, int_n_capacity, float_n_capacity)
            @test size(particle_system_base.n_particles_) == (1,)
            @test size(particle_system_base.is_alive_) == (n_capacity,)
            @test size(particle_system_base.is_movable_) == (n_capacity,)
            @test size(particle_system_base.cell_index_) == (n_capacity,)
            @test size(particle_system_base.int_properties_) == (n_capacity, int_n_capacity)
            @test size(particle_system_base.float_properties_) == (n_capacity, float_n_capacity)
        end
    end
    if USING_CUDA == true
        @testset "ParticleSystemBase CUDA" begin
            include("../../Head/cuda_test_head.jl")
            n_capacity = 100
            int_n_capacity = 20
            float_n_capacity = 30
            particle_system_base =
                EtherParallelParticles.Class.ParticleSystemBase(parallel, n_capacity, int_n_capacity, float_n_capacity)
            @test size(particle_system_base.n_particles_) == (1,)
            @test size(particle_system_base.is_alive_) == (n_capacity,)
            @test size(particle_system_base.is_movable_) == (n_capacity,)
            @test size(particle_system_base.cell_index_) == (n_capacity,)
            @test size(particle_system_base.int_properties_) == (n_capacity, int_n_capacity)
            @test size(particle_system_base.float_properties_) == (n_capacity, float_n_capacity)
        end
    end
    if USING_ONEAPI == true
        @testset "ParticleSystemBase ONEAPI" begin
            include("../../Head/oneapi_test_head.jl")
            n_capacity = 100
            int_n_capacity = 20
            float_n_capacity = 30
            particle_system_base =
                EtherParallelParticles.Class.ParticleSystemBase(parallel, n_capacity, int_n_capacity, float_n_capacity)
            @test size(particle_system_base.n_particles_) == (1,)
            @test size(particle_system_base.is_alive_) == (n_capacity,)
            @test size(particle_system_base.is_movable_) == (n_capacity,)
            @test size(particle_system_base.cell_index_) == (n_capacity,)
            @test size(particle_system_base.int_properties_) == (n_capacity, int_n_capacity)
            @test size(particle_system_base.float_properties_) == (n_capacity, float_n_capacity)
        end
    end
end
