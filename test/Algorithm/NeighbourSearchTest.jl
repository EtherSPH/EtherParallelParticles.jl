#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/21 15:45:53
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "NeighbourSearch" begin
    @testset "NeighbourSearch 2D $DEVICE" begin
        domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
        active_pair = [1 => 1, 1 => 2, 2 => 1]
        periodic_boundary_policy = EtherParallelParticles.Class.NonePeriodicBoundaryPolicy
        neighbour_system =
            EtherParallelParticles.Class.NeighbourSystem(parallel, domain, active_pair, periodic_boundary_policy)
        dim = 2
        neighbour_count = 50
        n_particles = 9
        int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = 1 * neighbour_count)
        float_named_tuple = (RVec = dim, nRVec = dim * neighbour_count, nR = neighbour_count)
        parameters = (c_0 = 340.0, gamma = 7, mu = 1e-3)
        capacityExpand(n)::typeof(n) = n
        d_particle_system = EtherParallelParticles.Class.ParticleSystem(
            parallel,
            domain,
            n_particles,
            int_named_tuple,
            float_named_tuple,
            parameters;
            capacityExpand = capacityExpand,
            basic_index_map_dict = Dict{Symbol, Symbol}(
                :PositionVec => :RVec,
                # :Tag => :Tag,
                :nCount => :nCount,
                :nIndex => :nIndex,
                # :nRVec => :nRVec,
                :nR => :nR,
            ),
        )
        h_particle_system = EtherParallelParticles.Class.mirror(d_particle_system)
        start_x = EtherParallelParticles.Class.get_first_x(domain)
        start_y = EtherParallelParticles.Class.get_first_y(domain)
        last_x = EtherParallelParticles.Class.get_last_x(domain)
        last_y = EtherParallelParticles.Class.get_last_y(domain)
        gap_x = EtherParallelParticles.Class.get_gap_x(domain)
        gap_y = EtherParallelParticles.Class.get_gap_y(domain)
        gap = EtherParallelParticles.Class.get_gap(domain)
        xy = zeros(FT, 9, 2)
        err = 1e-3
        xy[1, :] .= [start_x + err, start_y + err]
        xy[2, :] .= [start_x + gap / 2, start_y + err]
        xy[3, :] .= [start_x + gap * 3 / 2 - err, start_y + err]
        xy[4, :] .= [start_x + err, start_y + gap / 2]
        xy[5, :] .= [start_x + gap / 2, start_y + gap / 2]
        xy[6, :] .= [start_x + gap * 3 / 2 - err, start_y + gap / 2]
        xy[7, :] .= [start_x + err, start_y + gap - err]
        xy[8, :] .= [start_x + gap / 2, start_y + gap - err]
        xy[9, :] .= [start_x + gap * 3 / 2 - err, start_y + gap - err]
        for i in 1:9
            h_particle_system.base_.float_properties_[i, h_particle_system.basic_index_.PositionVec] = xy[i, 1]
            h_particle_system.base_.float_properties_[i, h_particle_system.basic_index_.PositionVec + 1] = xy[i, 2]
            h_particle_system.base_.is_alive_[i] = 1
            h_particle_system.base_.int_properties_[i, h_particle_system.basic_index_.Tag] = 1
        end
        EtherParallelParticles.Class.asyncto!(d_particle_system.base_, h_particle_system.base_)
        EtherParallelParticles.Algorithm.search!(d_particle_system, domain, neighbour_system)
        EtherParallelParticles.Algorithm.dynamic_search!(d_particle_system, domain, neighbour_system)
        EtherParallelParticles.Class.asyncto!(h_particle_system.base_, d_particle_system.base_)
        @test h_particle_system.base_.int_properties_[:, h_particle_system.basic_index_.nCount] ==
              [4, 5, 3, 5, 6, 3, 4, 5, 3]
        @test sort(
            h_particle_system.base_.int_properties_[
                1,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    1,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([2, 4, 5, 7])
        @test sort(
            h_particle_system.base_.int_properties_[
                2,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    2,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([1, 4, 5, 8, 3])
        @test sort(
            h_particle_system.base_.int_properties_[
                3,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    3,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([2, 6, 9])
        @test sort(
            h_particle_system.base_.int_properties_[
                4,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    4,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([1, 2, 5, 7, 8])
        @test sort(
            h_particle_system.base_.int_properties_[
                5,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    5,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([1, 2, 4, 7, 8, 6])
        @test sort(
            h_particle_system.base_.int_properties_[
                6,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    6,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([5, 3, 9])
        @test sort(
            h_particle_system.base_.int_properties_[
                7,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    7,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([1, 4, 5, 8])
        @test sort(
            h_particle_system.base_.int_properties_[
                8,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    8,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([2, 4, 5, 7, 9])
        @test sort(
            h_particle_system.base_.int_properties_[
                9,
                (h_particle_system.basic_index_.nIndex):(h_particle_system.basic_index_.nIndex + h_particle_system.base_.int_properties_[
                    9,
                    h_particle_system.basic_index_.nCount,
                ] - 1),
            ],
        ) == sort([8, 3, 6])
    end
end
