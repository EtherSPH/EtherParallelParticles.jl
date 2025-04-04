#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/08 21:56:03
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function iKernelFilter!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    w::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @wv(@i) += @float(w) * @vol(@j)
    @inbounds @wv_rho(@i) += @float(w) * @mass(@j)
    return nothing
end

@inline function sKernelFilter!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    w0::Real = 0,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @wv(@i) += @float(w0) * @vol(@i)
    @inbounds @wv_rho(@i) += @float(w0) * @mass(@i)
    @inbounds @rho(@i) = @wv_rho(@i) / @wv(@i)
    @wv(@i) = @float 0.0 # reset to zero
    @wv_rho(@i) = @float 0.0 # reset to zero
    return nothing
end
