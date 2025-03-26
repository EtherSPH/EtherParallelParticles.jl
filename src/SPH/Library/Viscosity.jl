#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/08 21:15:02
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function iClassicViscosity!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dw::Real = 0,
    mu::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    force::@float() =
        2 * @float(mu) * @mass(@j) * @r(@ij) * @float(dw) / (@rho(@i) * @rho(@j) * avoidzero(@r(@ij), 1 / @hinv(@ij)))
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @du(@i, i) += force * (@u(@i, i) - @u(@j, i))
    end
    return nothing
end

@inline function sArtificialViscosity!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    dw::Real = 0,
    alpha::Real = 0.1,
    beta::Real = 0.1,
    c::Real = 0.0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    v_dot_x::@float() = vdotx(@inter_args)
    if v_dot_x > 0 # which means that i and j are departing
        return nothing
    end
    rho::@float() = Math.Mean.arithmetic(@rho(@i), @rho(@j))
    phi::@float() = v_dot_x / (@hinv(@ij) * avoidzero(@r(@ij), 1 / @hinv(@ij)))
    force::@float() = (-@float(alpha) * @float(c) + @float(beta) * phi) * phi / rho
    force *= -@mass(@j) / (@rho(@i) * @rho(@j) * @r(@ij)) * @float(dw)
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @du(@i, i) += force * @rvec(@ij, i)
    end
    return nothing
end
