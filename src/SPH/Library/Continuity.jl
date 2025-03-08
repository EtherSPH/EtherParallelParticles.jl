#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 21:01:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function vdotx(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
)::eltype(FP) where {N, Dimension <: AbstractDimension{N}}
    v_dot_x::@float() = @float 0.0
    for i::@int() in 0:(N - 1)
        @inbounds v_dot_x += @rvec(NI, i) * (@u(@i, i) - @u(@j, i))
    end
    return v_dot_x
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
