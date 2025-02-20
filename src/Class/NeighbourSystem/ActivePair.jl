#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/12 00:38:05
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractActivePair{IT <: Integer} end

struct ActivePair{IT <: Integer} <: AbstractActivePair{IT}
    pair_vector_::Vector{Pair{IT, IT}}
    adjacency_matrix_::AbstractArray{IT, 2}
end

@inline function ActivePair(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    pair_vector::Vector{<:Pair{<:Integer, <:Integer}},
)::ActivePair{IT} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    active_pair_pair_vector = Vector{Pair{IT, IT}}()
    maximum_tag = IT(1)
    for pair in pair_vector
        push!(active_pair_pair_vector, Pair{IT, IT}(IT(pair.first), IT(pair.second)))
        maximum_tag = max(maximum_tag, pair.first, pair.second)
    end
    active_pair_adjacency_matrix = zeros(IT, maximum_tag, maximum_tag)
    for pair in active_pair_pair_vector
        active_pair_adjacency_matrix[pair.first, pair.second] = IT(1)
    end
    active_pair_adjacency_matrix = parallel(active_pair_adjacency_matrix)
    return ActivePair{IT}(active_pair_pair_vector, active_pair_adjacency_matrix)
end

function Base.show(io::IO, active_pair::ActivePair{IT}) where {IT <: Integer}
    println(io, "ActivePair{$IT}(")
    println(io, "    pair_vector: $(active_pair.pair_vector_)")
    println(io, "    adjacency_matrix: $(active_pair.adjacency_matrix_)")
    println(io, ")")
end
