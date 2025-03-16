#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/08 22:36:48
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

"""
# `accelerate!`

require:
- `VelocityVec`
- `dVelocityVec`
"""
@inline function accelerate!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dt::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @u(@i, i) += @du(@i, i) * @float(dt)
        @inbounds @du(@i, i) = @float 0.0
    end
    return nothing
end

"""
# `move!`

require:
- `PositionVec`
- `VelocityVec`
"""
@inline function move!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dt::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds for i::eltype(IP) in 0:(N - 1)
        @inbounds @x(@i, i) += @u(@i, i) * @float(dt)
    end
    return nothing
end

"""
# `accelerate_move!`

require:
- `VelocityVec`
- `dVelocityVec`
- `PositionVec`
"""
@inline function accelerate_move!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dt::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds for i::eltype(IP) in 0:(N - 1)
        @inbounds @x(@i, i) += (@u(@i, i) + @du(@i, i) * @float(dt) * @float(0.5)) * @float(dt)
        @inbounds @u(@i, i) += @du(@i, i) * @float(dt)
        @inbounds @du(@i, i) = @float 0.0
    end
    return nothing
end
