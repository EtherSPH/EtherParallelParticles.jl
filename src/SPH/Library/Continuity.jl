#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 21:01:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function sContinuity!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dt::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @rho(@i) += @drho(@i) * @float(dt)
    @inbounds @drho(@i) = @float 0.0
    return nothing
end

@inline function iClassicContinuity!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dw::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @drho(@i) += @mass(@j) * vdotx(@inter_args) * @float(dw) / @r(@ij)
    return nothing
end

@inline function iBalancedContinuity!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dw::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @drho(@i) += @rho(@i) * @vol(@j) * vdotx(@inter_args) * @float(dw) / @r(@ij)
    return nothing
end
