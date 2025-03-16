#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 21:01:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function continuity!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dt::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @rho(I) += @drho(I) * @float(dt)
    @inbounds @rho(I) = @float 0.0
    return nothing
end

@inline function classicContinuity!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @drho(@i) += @mass(@j) * vdotx(@inter_args) * @dw(@ij) / @r(@ij)
    return nothing
end

@inline function balancedContinuity!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @drho(@i) += @rho(@j) * @vol(@j) * vdotx(@inter_args) * @dw(@ij) / @dw(@ij)
    return nothing
end
