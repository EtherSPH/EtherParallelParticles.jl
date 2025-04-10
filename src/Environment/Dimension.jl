#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 19:48:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

@inline function dimension(::Type{Dimension}) where {N, Dimension <: AbstractDimension{N}}
    return N
end

@inline vector(::Val{Dimension}) where {N, Dimension <: AbstractDimension{N}} = N
@inline matrix(::Val{Dimension}) where {N, Dimension <: AbstractDimension{N}} = N * N
