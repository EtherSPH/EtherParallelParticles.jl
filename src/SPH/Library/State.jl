#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 20:30:50
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

"""
# `volume!`

require:
- `Mass`
- `Density`
- `Volume`
"""
@inline function volume!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @vol(@i) = @mass(@i) / @rho(@i)
    return nothing
end

@inline function acceleration!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    # copy `du` to `a`
    @inbounds for i::eltype(IP) in 0:(N - 1)
        @inbounds @a(@i, i) = @du(@i, i)
    end
    return nothing
end

"""
# `gravity!`

require:
- `dVelocityVec`
"""
@inline function gravity!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    gx::Real = 0,
    gy::Real = 0,
)::Nothing where {Dimension <: AbstractDimension{2}}
    @inbounds @du(@i, 0) += @float gx
    @inbounds @du(@i, 1) += @float gy
    return nothing
end

"""
# `gravity!`

require:
- `dVelocityVec`
"""
@inline function gravity!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    gx::Real = 0,
    gy::Real = 0,
    gz::Real = 0,
)::Nothing where {Dimension <: AbstractDimension{3}}
    @inbounds @du(@i, 0) += @float gx
    @inbounds @du(@i, 1) += @float gy
    @inbounds @du(@i, 2) += @float gz
    return nothing
end
