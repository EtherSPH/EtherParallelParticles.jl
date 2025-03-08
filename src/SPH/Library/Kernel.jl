#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/07 18:06:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function value!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @w(@ij) = Kernel.value(@r(@ij), Math.Mean.arithmetic(@h(@i), @h(@j)), kernel)
    return nothing
end

@inline function value!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    h_inv::Real,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @w(@ij) = Kernel._value(@r(@ij), @float(h_inv), kernel)
    return nothing
end

@inline function gradient!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @dw(@ij) = Kernel.gradient(@r(@ij), Math.Mean.arithmetic(@h(@i), @h(@j)), kernel)
    return nothing
end

@inline function gradient!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    h_inv::Real,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @dw(@ij) = Kernel._gradient(@r(@ij), @float(h_inv), kernel)
    return nothing
end

@inline function value_gradient!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    h_inv::@float() = Math.Mean.invharmonic(@h(@i), @h(@j))
    @inbounds @hinv(@ij) = h_inv
    @inbounds @w(@ij) = Kernel._value(@r(@ij), h_inv, kernel)
    @inbounds @dw(@ij) = Kernel._gradient(@r(@ij), h_inv, kernel)
    return nothing
end
