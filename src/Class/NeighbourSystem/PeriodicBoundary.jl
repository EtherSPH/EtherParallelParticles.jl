#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/14 16:31:48
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractPeriodicBoundaryPolicy end

struct NonePeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy end
abstract type HavePeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy end
struct PeriodicBoundaryPolicy2D{X, Y} <: HavePeriodicBoundaryPolicy end
struct PeriodicBoundaryPolicy3D{X, Y, Z} <: HavePeriodicBoundaryPolicy end

const PeriodicBoundaryPolicy2DAlongX = PeriodicBoundaryPolicy2D{true, false}
const PeriodicBoundaryPolicy2DAlongY = PeriodicBoundaryPolicy2D{false, true}
const PeriodicBoundaryPolicy2DAlongXY = PeriodicBoundaryPolicy2D{true, true}

const PeriodicBoundaryPolicy3DAlongX = PeriodicBoundaryPolicy3D{true, false, false}
const PeriodicBoundaryPolicy3DAlongY = PeriodicBoundaryPolicy3D{false, true, false}
const PeriodicBoundaryPolicy3DAlongZ = PeriodicBoundaryPolicy3D{false, false, true}
const PeriodicBoundaryPolicy3DAlongXY = PeriodicBoundaryPolicy3D{true, true, false}
const PeriodicBoundaryPolicy3DAlongYZ = PeriodicBoundaryPolicy3D{false, true, true}
const PeriodicBoundaryPolicy3DAlongZX = PeriodicBoundaryPolicy3D{true, false, true}
const PeriodicBoundaryPolicy3DAlongXYZ = PeriodicBoundaryPolicy3D{true, true, true}

abstract type AbstractPeriodicBoundary{FT <: AbstractFloat, PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy} end

struct PeriodicBoundary{FT <: AbstractFloat, PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy} <:
       AbstractPeriodicBoundary{FT, PeriodicBoundaryPolicy}
    neighbour_cell_relative_position_list_::AbstractArray{FT, 3}
end

@inline function PeriodicBoundary(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    ::AbstractDomain{IT, FT, Dimension},
    periodic_boundary_policy::Type{NonePeriodicBoundaryPolicy};
)::PeriodicBoundary{
    FT,
    NonePeriodicBoundaryPolicy,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    neighbour_cell_relative_position_list = parallel(zeros(FT, 1, 1, 1))
    return PeriodicBoundary{FT, NonePeriodicBoundaryPolicy}(neighbour_cell_relative_position_list)
end

@inline function PeriodicBoundary(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    periodic_boundary_policy::Type{<:HavePeriodicBoundaryPolicy};
)::PeriodicBoundary{
    FT,
    periodic_boundary_policy,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    n_cells = get_n(domain)
    # TODO: complete the neighbour_cell_relative_position_list in HavePeriodicBoundaryPolicy
    neighbour_cell_relative_position_list = parallel(zeros(FT, n_cells, neighbourCellCount(N), N))
    return PeriodicBoundary{FT, periodic_boundary_policy}(neighbour_cell_relative_position_list)
end

function Base.show(
    io::IO,
    periodic_boundary::PeriodicBoundary{FT, PeriodicBoundaryPolicy},
) where {FT <: AbstractFloat, PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy}
    println(io, "PeriodicBoundary{$FT, $PeriodicBoundaryPolicy}(")
    println(
        io,
        "    shape of `neighbour_cell_relative_position_list`: $(size(periodic_boundary.neighbour_cell_relative_position_list_))",
    )
    println(io, ")")
end
