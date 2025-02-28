#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/25 16:08:13
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * almost reduce no performance loss for complex type
 =#

include("../../cuda_head.jl")

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

@inline function broadcast!(
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

@inline function broadcast!(
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

@kernel function device_apply!(Dimension, x)
    I = @index(Global)
    broadcast!(x -> x + 1, Unary{Dimension, Vec}, I, 1, x)
    broadcast!(x -> x + 1, Unary{Dimension, Mat}, I, 1, x)
end

@inline function host_apply!(Dimension, x)
    device_apply!(Backend, 256)(Dimension, x, ndrange = (size(x, 1),))
    KernelAbstractions.synchronize(Backend)
end

@kernel function device_primitive!(Dimension, x)
    I = @index(Global)
    @inbounds x[I, 1] += 1
    @inbounds x[I, 2] += 1
    @inbounds x[I, 1] += 1
    @inbounds x[I, 2] += 1
    @inbounds x[I, 3] += 1
    @inbounds x[I, 4] += 1
end

@inline function host_primitive!(Dimension, x)
    device_apply!(Backend, 256)(Dimension, x, ndrange = (size(x, 1),))
    KernelAbstractions.synchronize(Backend)
end

a = randn(FT, 1024 * 1024, 4) |> CT
@info "warm up"
host_apply!(Dimension2D, a)
host_primitive!(Dimension2D, a)

@info "start benchmark for complex type"
@time begin
    for _ in 1:1_000
        host_apply!(Dimension2D, a)
    end
end

@info "start benchmark for primitive type"
@time begin
    for _ in 1:1_000
        host_primitive!(Dimension2D, a)
    end
end

# on MX-250 GPU
# [ Info: warm up
# [ Info: start benchmark for complex type
#   2.300813 seconds (41.05 k allocations: 1.331 MiB)
# [ Info: start benchmark for primitive type
#   2.304729 seconds (41.00 k allocations: 1.328 MiB)
