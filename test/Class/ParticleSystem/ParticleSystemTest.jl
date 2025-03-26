#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/18 15:08:13
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "ParticleSystem" begin
    include("ParticleSystemBaseTest.jl")
    include("NamedIndexTest.jl")
    @testset "ParticleSystem $DEVICE" begin
        domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
        dim = 2
        neighbour_count = 50
        n_particles = 100
        int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = 1 * neighbour_count)
        float_named_tuple = (
            PositionVec = dim,
            Mass = 1,
            Density = 1,
            Volume = 1,
            VelocityVec = dim,
            dVelocityVec = dim,
            dDensity = 1,
            Pressure = 1,
            StrainMat = dim * dim,
            dStrainMat = dim * dim,
            StressMat = dim * dim,
            nRVec = dim * neighbour_count,
            nR = neighbour_count,
            nW = neighbour_count,
            nDW = neighbour_count,
        )
        capacityExpand(n)::typeof(n) = n + 100
        particle_system = EtherParallelParticles.Class.ParticleSystem(
            parallel,
            domain,
            n_particles,
            int_named_tuple,
            float_named_tuple,
            (c_0 = 340.0, gamma = 7, mu = 1e-3);
            capacityExpand = capacityExpand,
            basic_index_map_dict = Dict{Symbol, Symbol}(
                :PositionVec => :PositionVec,
                # :Tag => :Tag,
                :nCount => :nCount,
                :nIndex => :nIndex,
                # :nRVec => :nRVec,
                :nR => :nR,
            ),
        )
        @test EtherParallelParticles.Class.get_n_particles(particle_system) == n_particles
        @test EtherParallelParticles.Class.get_n_capacity(particle_system) == capacityExpand(n_particles)
        @test EtherParallelParticles.Class.get_alive_n_particles(particle_system) == n_particles
        @test EtherParallelParticles.Class.get_n_int_capacity(particle_system) == 1 + 1 + 1 + 1 * neighbour_count
        @test EtherParallelParticles.Class.get_n_float_capacity(particle_system) ==
              dim +
              1 +
              1 +
              1 +
              dim +
              dim +
              1 +
              1 +
              dim * dim +
              dim * dim +
              dim * dim +
              dim * neighbour_count +
              neighbour_count +
              neighbour_count +
              neighbour_count
        mirror_ps = Class.mirror(particle_system)
        Class.syncto!(mirror_ps, particle_system)
        Class.asyncto!(mirror_ps, particle_system)
        Class.syncto!(particle_system, mirror_ps)
        Class.asyncto!(particle_system, mirror_ps)
        @test typeof(mirror_ps.base_.int_properties_) == Array{IT, 2}
        @test typeof(mirror_ps.base_.float_properties_) == Array{FT, 2}
        @test typeof(particle_system.base_.int_properties_) <: CT
        @test typeof(particle_system.base_.float_properties_) <: CT
        @test typeof(particle_system.parameters_.c_0) == FT
        @test typeof(particle_system.parameters_.gamma) == IT
        @test typeof(particle_system.parameters_.mu) == FT
        @test particle_system.basic_index_.Tag == 1
        @test particle_system.basic_index_.IsMovable == 2
        @test particle_system.basic_index_.nCount == 3
        @test particle_system.basic_index_.nIndex == 4
        @test particle_system.basic_index_.nR == 124
        @test particle_system.basic_index_.PositionVec == 1
        @test particle_system.basic_index_.nRVec == 24
        @test particle_system.basic_index_.Tag == 1
        Class.set_n_particles!(particle_system, 200)
        @test EtherParallelParticles.Class.get_n_particles(particle_system) == 200
    end
end
