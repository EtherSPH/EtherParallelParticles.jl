#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/24 20:50:22
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * this works!
 =#

include("../../oneapi_head.jl")

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

abstract type AbstractTensorOperation{N, Dimension <: AbstractDimension} end

abstract type AbstractVectorOperation{Dimension <: AbstractDimension} <: AbstractTensorOperation{1, Dimension} end
struct Vec{Dimension <: AbstractDimension} <: AbstractVectorOperation{Dimension} end

abstract type AbstractMatrixOperation{Dimension <: AbstractDimension} <: AbstractTensorOperation{2, Dimension} end
struct Mat{Dimension <: AbstractDimension} <: AbstractMatrixOperation{Dimension} end

@inline function unary(f::Function, row::Integer, column::Integer, data)
    @inbounds return f(data[row, column])
end

@inline function unary!(f::Function, row::Integer, column::Integer, data)
    @inbounds data[row, column] = f(data[row, column])
    return nothing
end

@inline function unary!(
    f::Function,
    ::Type{Vec{Dimension}},
    row_1::Integer,
    column_1::Integer,
    row_2::Integer,
    column_2::Integer,
    data,
) where {Dimension <: AbstractDimension{2}}
    @inbounds data[row_1, column_1] = f(data[row_2, column_2])
    @inbounds data[row_1, column_1 + 1] = f(data[row_2, column_2 + 1])
    return nothing
end

@inline function unary!(
    f::Function,
    ::Type{Vec{Dimension}},
    row_1::Integer,
    column_1::Integer,
    column_2::Integer,
    data,
) where {Dimension <: AbstractDimension{2}}
    @inbounds data[row_1, column_1] = f(data[row_1, column_2])
    @inbounds data[row_1, column_1 + 1] = f(data[row_1, column_2 + 1])
    return nothing
end

@inline function unary!(
    f::Function,
    ::Type{Vec{Dimension}},
    row_1::Integer,
    column_1::Integer,
    data,
) where {Dimension <: AbstractDimension{2}}
    @inbounds data[row_1, column_1] = f(data[row_1, column_1])
    @inbounds data[row_1, column_1 + 1] = f(data[row_1, column_1 + 1])
    return nothing
end

@kernel function device_unary!(Dimension::Type{SomeDimension}, data) where {N, SomeDimension <: AbstractDimension{N}}
    I = @index(Global)
    unary!(x -> x + 1, Vec{Dimension}, I, 1, data)
end

a = rand(FT, 2, 2) |> CT
@info a
kernel = device_unary!(Backend, 2)
kernel(Dimension2D, a, ndrange = (2,))
KernelAbstractions.synchronize(Backend)
@info a
