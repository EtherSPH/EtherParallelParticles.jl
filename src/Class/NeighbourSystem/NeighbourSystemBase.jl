#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/11 15:08:42
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractNeighbourSystemBase{IT <: Integer} end

struct NeighbourSystemBase{IT <: Integer} <: AbstractNeighbourSystemBase{IT}
    contained_particle_index_count_::AbstractArray{IT, 1} # (n_cells, )
    contained_particle_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
    # ! including the cell itself, this field is the only field need `atomic operation`
    neighbour_cell_index_count_::AbstractArray{IT, 1} # (n_cells, )
    neighbour_cell_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
end

function Base.show(io::IO, neighbour_system::NeighbourSystemBase{IT}) where {IT <: Integer}
    println(io, "NeighbourSystemBase{$IT}(")
    println(io, "    number of cells: $(length(neighbour_system.contained_particle_index_count_))")
    println(io, "    max number of contained particles: $(size(neighbour_system.contained_particle_index_list_, 2))")
    println(io, "    max number of neighbour cells: $(size(neighbour_system.neighbour_cell_index_list_, 2))")
    println(io, ")")
end

@kernel function device_initializeNeighbourSystem!(
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    I::IT = @index(Global)
    i::IT, j::IT = indexLinearToCartesian(domain, I)
    n_x::IT = get_n_x(domain)
    n_y::IT = get_n_y(domain)
    for di::IT in -1:1
        ii::IT = i + di
        for dj::IT in -1:1
            jj::IT = j + dj
            if ii >= 1 && ii <= n_x && jj >= 1 && jj <= n_y
                @inbounds neighbour_cell_index_count[I] += 1
                @inbounds neighbour_cell_index_list[I, neighbour_cell_index_count[I]] =
                    indexCartesianToLinear(domain, ii, jj)
            end
        end
    end
end

@kernel function device_initializeNeighbourSystem!(
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{3}}
    I::IT = @index(Global)
    # TODO: add 3D support
end

@inline function host_initializeNeighbourSystem!(
    ::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    device_initializeNeighbourSystem!(Backend, n_threads)(
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
        ndrange = (get_n(domain),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function NeighbourSystemBase(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::NeighbourSystemBase{
    IT,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    n_cells = get_n(domain)
    contained_particle_index_count = parallel(zeros(IT, n_cells))
    contained_particle_index_list = parallel(zeros(IT, n_cells, max_neighbour_number))
    neighbour_cell_index_count = parallel(zeros(IT, n_cells))
    neighbour_cell_count = IT(neighbourCellCount(N))
    neighbour_cell_index_list = parallel(zeros(IT, n_cells, neighbour_cell_count))
    host_initializeNeighbourSystem!(
        parallel,
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list;
        n_threads = n_threads,
    )
    return NeighbourSystemBase{IT}(
        contained_particle_index_count,
        contained_particle_index_list,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
    )
end
