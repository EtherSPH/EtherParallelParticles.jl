#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/07 18:06:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function iValue(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::@float() where {N, Dimension <: AbstractDimension{N}}
    return Kernel.value(@r(@ij), Math.Mean.arithmetic(@h(@i), @h(@j)), kernel)
end

@inline function iValue(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    hinv::Real,
    kernel::AbstractKernel;
)::@float() where {N, Dimension <: AbstractDimension{N}}
    return Kernel._value(@r(@ij), hinv, kernel)
end

@inline function iValue!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @w(@ij) = iValue(@inter_args, kernel)
    return nothing
end

@inline function iValue!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    hinv::Real,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @w(@ij) = iValue(@inter_args, hinv, kernel)
    return nothing
end

@inline function iGradient(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::@float() where {N, Dimension <: AbstractDimension{N}}
    return Kernel.gradient(@r(@ij), Math.Mean.arithmetic(@h(@i), @h(@j)), kernel)
end

@inline function iGradient(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    hinv::Real,
    kernel::AbstractKernel;
)::@float() where {N, Dimension <: AbstractDimension{N}}
    return Kernel._gradient(@r(@ij), hinv, kernel)
end

@inline function iGradient!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @dw(@ij) = iGradient(@inter_args, kernel)
    return nothing
end

@inline function iGradient!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    hinv::Real,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds @dw(@ij) = iGradient(@inter_args, hinv, kernel)
    return nothing
end

@inline function iValueGradient!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    hinv::@float() = Math.Mean.invharmonic(@h(@i), @h(@j))
    @inbounds @hinv(@ij) = hinv
    @inbounds @w(@ij) = Kernel._value(@r(@ij), hinv, kernel)
    @inbounds @dw(@ij) = Kernel._gradient(@r(@ij), hinv, kernel)
    return nothing
end
