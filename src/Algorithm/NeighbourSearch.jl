#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/18 22:04:22
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

# * ==================== insertParticlesIntoCells begin ==================== * #

@kernel function device_insertParticlesIntoCells!(
    domain_2d::AbstractDomain{IT, FT, Dimension2D},
    ps_is_alive,
    @Const(ps_is_movable),
    ps_cell_index,
    @Const(ps_float_properties),
    ns_contained_particle_index_count,
    ns_contained_particle_index_list,
    index_PositionVec::IT,
) where {IT <: Integer, FT <: AbstractFloat}
    I::IT = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        # case here:
        # 1. movable: cell index must be calculated again
        # 2. immovable: if cell_index == 0, then calculate cell index
        #               if cell_index != 0, cell index does not need to be calculated
        if ps_is_movable[I] == 0 && ps_cell_index[I] != 0
            particle_in_cell_index = Atomix.@atomic ns_contained_particle_index_count[cell_index] += 1
            @inbounds ns_contained_particle_index_list[cell_index, particle_in_cell_index] = I
        else
            @inbounds x::FT = ps_float_properties[I, index_PositionVec]
            @inbounds y::FT = ps_float_properties[I, index_PositionVec + 1]
            if Class.inside(domain_2d, x, y)
                cell_index::IT = Class.indexLinearFromPosition(domain_2d, x, y)
                @inbounds ps_cell_index[I] = cell_index
                particle_in_cell_index = Atomix.@atomic ns_contained_particle_index_count[cell_index] += 1
                @inbounds ns_contained_particle_index_list[cell_index, particle_in_cell_index] = I
            else
                @inbounds ps_is_alive[I] = 0
                @inbounds ps_cell_index[I] = 0
            end
        end
    end
end

# TODO: add 3D support

@inline function host_insertParticlesIntoCells!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy},
    n_particles::IT;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    device_insertParticlesIntoCells!(Backend, n_threads)(
        domain,
        particle_system.device_base_.is_alive_,
        particle_system.device_base_.is_movable_,
        particle_system.device_base_.cell_index_,
        particle_system.device_base_.float_properties_,
        neighbour_system.base_.contained_particle_index_count_,
        neighbour_system.base_.contained_particle_index_list_,
        particle_system.basic_index_.PositionVec,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

# * ==================== insertParticlesIntoCells end ==================== * #

# * ==================== findNeighbourParticlesFromCells begin ==================== * #

@inline function applyPeriodicBoundaryPolicy(
    ::Type{NonePeriodicBoundaryPolicy},
    dx::FT,
    dy::FT,
    ::IT,
    ::IT,
    neighbour_cell_relative_position_list,
)::Tuple{FT, FT} where {FT <: AbstractFloat, IT <: Integer}
    return dx, dy
end

@inline function applyPeriodicBoundaryPolicy(
    ::Type{<:HavePeriodicBoundaryPolicy},
    dx::FT,
    dy::FT,
    cell_index::IT,
    i_neighbour_cell::IT,
    neighbour_cell_relative_position_list,
)::Tuple{FT, FT} where {FT <: AbstractFloat, IT <: Integer}
    @inbounds return dx + neighbour_cell_relative_position_list[cell_index, i_neighbour_cell, 1],
    dy + neighbour_cell_relative_position_list[cell_index, i_neighbour_cell, 2]
end

@kernel function device_findNeighbourParticlesFromCells!(
    domain_2d::AbstractDomain{IT, FT, Dimension2D},
    periodic_boundary_policy::Type{<:AbstractPeriodicBoundaryPolicy},
    @Const(ps_is_alive),
    @Const(ps_cell_index),
    ps_int_properties, # 2D array
    ps_float_properties, # 2D array
    ns_contained_particle_index_count, # 1D array
    ns_contained_particle_index_list, # 2D array
    ns_neighbour_cell_index_count, # 1D array
    ns_neighbour_cell_index_list, # 2D array
    ns_adjacency_matrix, # 2D array
    ns_neighbour_cell_relative_position_list, # 3D array
    index_Tag::IT,
    index_PositionVec::IT,
    index_nCount::IT,
    index_nIndex::IT,
    index_nRVec::IT,
    index_nR::IT,
) where {IT <: Integer, FT <: AbstractFloat}
    I::IT = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        @inbounds ps_int_properties[I, index_nCount] = 0
        @inbounds cell_index::IT = ps_cell_index[I]
        @inbounds n_neighbour_cells::IT = ns_neighbour_cell_index_count[cell_index]
        for i_neighbour_cell::IT in 1:n_neighbour_cells
            @inbounds neighbour_cell_index::IT = ns_neighbour_cell_index_list[cell_index, i_neighbour_cell]
            @inbounds n_particles_in_neighbour_cell::IT = ns_contained_particle_index_count[neighbour_cell_index]
            for i_particle::IT in 1:n_particles_in_neighbour_cell
                @inbounds J::IT = ns_contained_particle_index_list[neighbour_cell_index, i_particle]
                @inbounds if I != J &&
                             ns_adjacency_matrix[ps_int_properties[I, index_Tag], ps_int_properties[J, index_Tag]] == 1
                    @inbounds dx::FT =
                        ps_float_properties[I, index_PositionVec] - ps_float_properties[J, index_PositionVec]
                    @inbounds dy::FT =
                        ps_float_properties[I, index_PositionVec + 1] - ps_float_properties[J, index_PositionVec + 1]
                    dx, dy = applyPeriodicBoundaryPolicy(
                        periodic_boundary_policy,
                        dx,
                        dy,
                        cell_index,
                        i_neighbour_cell,
                        ns_neighbour_cell_relative_position_list,
                    )
                    dr_square::FT = dx * dx + dy * dy
                    if dr_square <= Class.get_gap_square(domain_2d)
                        @inbounds ps_int_properties[I, index_nCount] += 1
                        @inbounds neighbour_count::IT = ps_int_properties[I, index_nCount]
                        @inbounds ps_int_properties[I, index_nIndex + neighbour_count - 1] = J
                        @inbounds ps_float_properties[I, index_nRVec + 2 * (neighbour_count - 1)] = dx
                        @inbounds ps_float_properties[I, index_nRVec + 2 * neighbour_count - 1] = dy
                        @inbounds ps_float_properties[I, index_nR + neighbour_count - 1] = sqrt(dr_square)
                    end
                end
            end
        end
    end
end

# TODO: add 3D support

@inline function host_findNeighbourParticlesFromCells!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy},
    n_particles::IT;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    device_findNeighbourParticlesFromCells!(Backend, n_threads)(
        domain,
        PeriodicBoundaryPolicy,
        particle_system.device_base_.is_alive_,
        particle_system.device_base_.cell_index_,
        particle_system.device_base_.int_properties_,
        particle_system.device_base_.float_properties_,
        neighbour_system.base_.contained_particle_index_count_,
        neighbour_system.base_.contained_particle_index_list_,
        neighbour_system.base_.neighbour_cell_index_count_,
        neighbour_system.base_.neighbour_cell_index_list_,
        neighbour_system.active_pair_.adjacency_matrix_,
        neighbour_system.periodic_boundary_.neighbour_cell_relative_position_list_,
        particle_system.basic_index_.Tag,
        particle_system.basic_index_.PositionVec,
        particle_system.basic_index_.nCount,
        particle_system.basic_index_.nIndex,
        particle_system.basic_index_.nRVec,
        particle_system.basic_index_.nR,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

# * ==================== findNeighbourParticlesFromCells end ==================== * #

@inline function search!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy};
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    Class.clean!(neighbour_system)
    n_particles::IT = Class.get_n_particles(particle_system)
    host_insertParticlesIntoCells!(particle_system, domain, neighbour_system, n_particles; n_threads = n_threads)
    host_findNeighbourParticlesFromCells!(particle_system, domain, neighbour_system, n_particles; n_threads = n_threads)
    return nothing
end
