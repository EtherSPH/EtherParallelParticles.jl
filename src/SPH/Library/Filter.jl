#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/08 21:56:03
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function kernelFilter!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @wv(@i) += @w(@ij) * @vol(@j)
    @inbounds @wv_rho(@i) += @w(@ij) * @mass(@j)
    return nothing
end

@inline function kernelFilter!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    w::@float() = Kernel.value0(@h(@i), kernel)
    @inbounds @wv(@i) += w * @vol(@i)
    @inbounds @wv_rho(@i) += w * @mass(@i)
    @inbounds @rho(@i) += @wv_rho(@i) / @wv(@i)
    @wv(@i) = @float 0.0 # reset to zero
    @wv_rho(@i) = @float 0.0 # reset to zero
    return nothing
end
