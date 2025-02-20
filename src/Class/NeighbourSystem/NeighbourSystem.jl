#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/11 15:07:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function neighbourCellCount(dim::IT)::IT where {IT <: Integer}
    return 3^dim # 9 for 2D, 27 for 3D
end

include("NeighbourSystemBase.jl")
include("ActivePair.jl")
include("PeriodicBoundary.jl")

"""
    - `base_`: the base of neighbour system
    - `active_pair_`: the active pair of neighbour system
    - `periodic_boundary_`: the periodic boundary of neighbour system
"""
abstract type AbstractNeighbourSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
} end

@inline function get_n(
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy},
)::IT where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    return length(neighbour_system.base_.contained_particle_index_count_)
end

@inline function clean!(
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    KernelAbstractions.fill!(neighbour_system.base_.contained_particle_index_count_, IT(0))
    return nothing
end

struct NeighbourSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
} <: AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy}
    base_::NeighbourSystemBase{IT}
    active_pair_::ActivePair{IT}
    periodic_boundary_::PeriodicBoundary{FT, PeriodicBoundaryPolicy}
end

@inline function NeighbourSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    active_pair::Vector{<:Pair{<:Integer, <:Integer}},
    periodic_boundary_policy::Type{<:AbstractPeriodicBoundaryPolicy};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::NeighbourSystem{
    IT,
    FT,
    CT,
    Backend,
    periodic_boundary_policy,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    base = NeighbourSystemBase(parallel, domain; max_neighbour_number = max_neighbour_number, n_threads = n_threads)
    active_pair = ActivePair(parallel, active_pair)
    periodic_boundary = PeriodicBoundary(parallel, domain, periodic_boundary_policy)
    return NeighbourSystem{IT, FT, CT, Backend, periodic_boundary_policy}(base, active_pair, periodic_boundary)
end

function Base.show(
    io::IO,
    neighbour_system::NeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy},
) where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    println(io, "NeighbourSystem{$IT, $FT, $CT, $Backend, $PeriodicBoundaryPolicy}(")
    println(io, "    base: NeighbourSystemBase{$IT}")
    println(io, "        number of cells: $(get_n(neighbour_system))")
    println(
        io,
        "        max number of contained particles: $(size(neighbour_system.base_.contained_particle_index_list_, 2))",
    )
    println(io, "        max number of neighbour cells: $(size(neighbour_system.base_.neighbour_cell_index_list_, 2))")
    println(io, "    active pair: ActivePair{$IT}")
    println(io, "        pair_vector: $(neighbour_system.active_pair_.pair_vector_)")
    println(io, "        adjacency_matrix: $(neighbour_system.active_pair_.adjacency_matrix_)")
    println(io, "    periodic boundary: PeriodicBoundary{$FT, $PeriodicBoundaryPolicy}")
    println(
        io,
        "        shape of `neighbour_cell_relative_position_list`: $(size(neighbour_system.periodic_boundary_.neighbour_cell_relative_position_list_))",
    )
    println(io, ")")
end
