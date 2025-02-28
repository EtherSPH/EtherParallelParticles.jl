#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/27 15:56:39
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * almost don't reduce the performance for `x -> x + 1` and `plus(x) = x + 1` in the native function.
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

@inline capacity(::Val{1}, ::Val{__Vector}) = 1
@inline capacity(::Val{2}, ::Val{__Vector}) = 2
@inline capacity(::Val{3}, ::Val{__Vector}) = 3
@inline capacity(::Val{1}, ::Val{__Matrix}) = 1
@inline capacity(::Val{2}, ::Val{__Matrix}) = 4
@inline capacity(::Val{3}, ::Val{__Matrix}) = 9

abstract type AbstractOperator{Dimension <: AbstractDimension} end

struct Unary{Dimension <: AbstractDimension, Tensor <: AbstractTensor} <: AbstractOperator{Dimension} end
struct Binary{Dimension <: AbstractDimension, Tensor1 <: AbstractTensor, Tensor2 <: AbstractTensor} <:
       AbstractOperator{Dimension} end

@inline function broadcast!(__func::Function, n::Integer, row::Integer, column::Integer, data)::Nothing
    for i::typeof(column) in 0:(n - 1)
        @inbounds data[row, column + i] = __func(data[row, column + i])
    end
    return nothing
end

@inline function broadcast!(
    __func::Function,
    ::Type{Unary{Dimension, __Tensor}},
    row::Integer,
    column::Integer,
    data,
)::Nothing where {N, Dimension <: AbstractDimension{N}, __Tensor <: AbstractTensor}
    broadcast!(__func, capacity(Val(N), Val(__Tensor)), row, column, data)
    return nothing
end

@kernel function device_native_f!(data)
    I = @index(Global)
    broadcast!(x -> x + 1, Unary{Dimension2D, __Vector}, I, 1, data)
    broadcast!(x -> x + 1, Unary{Dimension3D, __Vector}, I, 1, data)
    broadcast!(x -> x + 1, Unary{Dimension2D, __Matrix}, I, 1, data)
end

@inline plus(x) = x + 1

@kernel function device_primitive_f!(data)
    I = @index(Global)
    broadcast!(plus, Unary{Dimension2D, __Vector}, I, 1, data)
    broadcast!(plus, Unary{Dimension3D, __Vector}, I, 1, data)
    broadcast!(plus, Unary{Dimension2D, __Matrix}, I, 1, data)
end

@inline function host_native_f!(data)
    device_native_f!(Backend, 256)(data, ndrange = (size(data, 1),))
    KernelAbstractions.synchronize(Backend)
end

@inline function host_primitive_f!(data)
    device_primitive_f!(Backend, 256)(data, ndrange = (size(data, 1),))
    KernelAbstractions.synchronize(Backend)
end

a = zeros(FT, 1024 * 1024, 4) |> CT
@info "warm up"
host_native_f!(a)
host_primitive_f!(a)

@info "benchmark start: `x -> x + 1`"
@time begin
    for _ in 1:1000
        host_native_f!(a)
    end
end
@info "benchmark start: `plus(x) = x + 1`"
@time begin
    for _ in 1:1000
        host_primitive_f!(a)
    end
end

# * on MX-250
# [ Info: warm up
# [ Info: benchmark start: `x -> x + 1`
#   3.397532 seconds (17.05 k allocations: 347.469 KiB, 14 lock conflicts)
# [ Info: benchmark start: `plus(x) = x + 1`
#   3.389487 seconds (17.02 k allocations: 344.266 KiB, 18 lock conflicts)

# [ Info: warm up
# [ Info: benchmark start: `x -> x + 1`
#   3.518050 seconds (17.00 k allocations: 343.688 KiB, 45 lock conflicts)
# [ Info: benchmark start: `plus(x) = x + 1`
#   3.389863 seconds (17.00 k allocations: 343.734 KiB, 12 lock conflicts)

# [ Info: warm up
# [ Info: benchmark start: `x -> x + 1`
#   3.430987 seconds (17.00 k allocations: 343.750 KiB, 18 lock conflicts)
# [ Info: benchmark start: `plus(x) = x + 1`
#   3.447400 seconds (17.00 k allocations: 343.719 KiB, 30 lock conflicts)
