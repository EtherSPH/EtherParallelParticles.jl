#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/26 01:36:46
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * broadcast!(f...) while f is `(x, y) -> -x + y` is allowed.
 =#

include("../../cuda_head.jl")

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

abstract type AbstractTensor{N} end

struct __Scalar <: AbstractTensor{0} end
struct __Vector <: AbstractTensor{1} end
struct __Matrix <: AbstractTensor{2} end

const Sca = __Scalar
const Vec = __Vector
const Mat = __Matrix

@inline __vector(::Val{1}) = 1
@inline __vector(::Val{2}) = 2
@inline __vector(::Val{3}) = 3

@inline __matrix(::Val{1}) = 1
@inline __matrix(::Val{2}) = 4
@inline __matrix(::Val{3}) = 9

abstract type AbstractOperator{Dimension <: AbstractDimension} end

struct Unary{Dimension <: AbstractDimension, Tensor <: AbstractTensor} <: AbstractOperator{Dimension} end
struct Binary{Dimension <: AbstractDimension, Tensor1 <: AbstractTensor, Tensor2 <: AbstractTensor} <:
       AbstractOperator{Dimension} end

@inline function broadcast!(f::Function, n::Integer, row::Integer, column::Integer, data)::Nothing
    for i::typeof(row) in 0:(n - 1)
        @inbounds data[row, column + i] = f(data[row, column + i])
    end
    return nothing
end

@inline function broadcast!(
    f::Function,
    ::Type{Unary{Dimension, __Vector}},
    row::Integer,
    column::Integer,
    data,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    broadcast!(f, N, row, column, data)
    return nothing
end

@inline function broadcast!(
    f::Function,
    ::Type{Unary{Dimension, __Matrix}},
    row::Integer,
    column::Integer,
    data,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    broadcast!(f, __matrix(Val(N)), row, column, data)
    return nothing
end

@inline function broadcast!(
    f::Function,
    n::Integer,
    row_1::Integer,
    column_1::Integer,
    row_2::Integer,
    column_2::Integer,
    data,
)::Nothing
    for i::typeof(row_1) in 0:(n - 1)
        @inbounds data[row_1, column_1 + i] = f(data[row_1, column_1 + i], data[row_2, column_2 + i])
    end
    return nothing
end

@inline function broadcast!(
    f::Function,
    ::Type{Binary{Dimension, __Vector, __Vector}},
    row_1::Integer,
    column_1::Integer,
    row_2::Integer,
    column_2::Integer,
    data,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    broadcast!(f, N, row_1, column_1, row_2, column_2, data)
    return nothing
end

@inline function broadcast!(
    f::Function,
    ::Type{Binary{Dimension, __Matrix, __Matrix}},
    row_1::Integer,
    column_1::Integer,
    row_2::Integer,
    column_2::Integer,
    data,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    broadcast!(f, __matrix(Val(N)), row_1, column_1, row_2, column_2, data)
    return nothing
end

@kernel function device_apply!(Dimension, x)
    I = @index(Global)
    broadcast!(a -> a + 1, Unary{Dimension, Vec}, I, 1, x)
    broadcast!(a -> a + 1, Unary{Dimension, Mat}, I, 1, x)
    broadcast!((a, b) -> -a + b, Binary{Dimension, Vec, Vec}, I, 1, I, 1, x)
    broadcast!((a, b) -> -a + b, Binary{Dimension, Mat, Mat}, I, 1, I, 1, x)
end

@inline function host_apply!(Dimension, x)
    device_apply!(Backend, 256)(Dimension, x, ndrange = (size(x, 1),))
    KernelAbstractions.synchronize(Backend)
end

a = randn(FT, 2, 4) |> CT
@info a
host_apply!(Dimension2D, a)
@info a
