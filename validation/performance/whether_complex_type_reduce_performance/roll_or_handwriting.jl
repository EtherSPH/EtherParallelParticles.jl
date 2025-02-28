#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/25 19:02:38
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using KernelAbstractions
using CUDA
using Random

const IT = Int32
const FT = Float32
const CT = CUDA.CuArray
const Backend = CUDA.CUDABackend()

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

abstract type AbstractTensor{N} end

struct Sca <: AbstractTensor{0} end
struct Vec <: AbstractTensor{1} end
struct Mat <: AbstractTensor{2} end

abstract type AbstractLinearOperator{Dimension <: AbstractDimension} end

struct Unary{Dimenion <: AbstractDimension, Tensor <: AbstractTensor} <: AbstractLinearOperator{Dimenion} end

@inline function hw_broadcast!(
    f::Function,
    ::Type{Unary{Dimension, Vec}},
    row::Integer,
    column::Integer,
    data,
)::Nothing where {Dimension <: AbstractDimension{2}}
    @inbounds data[row, column] = f(data[row, column])
    @inbounds data[row, column + 1] = f(data[row, column + 1])
    return nothing
end

@inline function hw_broadcast!(
    f::Function,
    ::Type{Unary{Dimension, Mat}},
    row::Integer,
    column::Integer,
    data,
)::Nothing where {Dimension <: AbstractDimension{2}}
    @inbounds data[row, column] = f(data[row, column])
    @inbounds data[row, column + 1] = f(data[row, column + 1])
    @inbounds data[row, column + 2] = f(data[row, column + 2])
    @inbounds data[row, column + 3] = f(data[row, column + 3])
    return nothing
end

@inline function roll_broadcast!(
    f::Function,
    ::Type{Unary{Dimension, Vec}},
    row::Integer,
    column::Integer,
    data,
)::Nothing where {Dimension <: AbstractDimension{2}}
    for i in 0:1
        @inbounds data[row, column + i] = f(data[row, column + i])
    end
    return nothing
end

@inline function roll_broadcast!(
    f::Function,
    ::Type{Unary{Dimension, Mat}},
    row::Integer,
    column::Integer,
    data,
)::Nothing where {Dimension <: AbstractDimension{2}}
    for i in 0:3
        @inbounds data[row, column + i] = f(data[row, column + i])
    end
    return nothing
end

@kernel function device_hw!(Dimension, x)
    I = @index(Global)
    hw_broadcast!(x -> x + 1, Unary{Dimension, Vec}, I, 1, x)
    hw_broadcast!(x -> x + 1, Unary{Dimension, Mat}, I, 1, x)
end

@inline function host_hw!(Dimension, x)
    device_hw!(Backend, 256)(Dimension, x, ndrange = (size(x, 1),))
    KernelAbstractions.synchronize(Backend)
end

@kernel function device_roll!(Dimension, x)
    I = @index(Global)
    roll_broadcast!(x -> x + 1, Unary{Dimension, Vec}, I, 1, x)
    roll_broadcast!(x -> x + 1, Unary{Dimension, Mat}, I, 1, x)
end

@inline function host_roll!(Dimension, x)
    device_roll!(Backend, 256)(Dimension, x, ndrange = (size(x, 1),))
    KernelAbstractions.synchronize(Backend)
end

a = randn(FT, 1024 * 1024, 4) |> CT
@info "warm up"
host_hw!(Dimension2D, a)
host_roll!(Dimension2D, a)

@info "start benchmark for handwriting"
@time begin
    for _ in 1:1_000
        host_hw!(Dimension2D, a)
    end
end

@info "start benchmark for roll"
@time begin
    for _ in 1:1_000
        host_roll!(Dimension2D, a)
    end
end
