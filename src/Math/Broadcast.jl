#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/25 17:27:27
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

struct broadcast!{Pattern <: AbstractPattern, N} <: AbstractOperator end

@inline function broadcast!{Unary{Tensor}, N}(
    __func::Function,
    row_1::Integer,
    column_1::Integer,
    row_2::Integer,
    column_2::Integer,
    data,
)::Nothing where {N, Tensor <: AbstractTensor}
    for i::typeof(column_1) in 0:(capacity(Val(Tensor), Val(N)) - 1)
        @inbounds data[row_1, column_1 + i] = __func(data[row_2, column_2 + i])
    end
    return nothing
end

@inline function broadcast!{Unary{Tensor}, N}(
    __func::Function,
    row_1::Integer,
    column_1::Integer,
    column_2::Integer,
    data,
)::Nothing where {N, Tensor <: AbstractTensor}
    for i::typeof(column_1) in 0:(capacity(Val(Tensor), Val(N)) - 1)
        @inbounds data[row_1, column_1 + i] = __func(data[row_1, column_2 + i])
    end
    return nothing
end

@inline function broadcast!{Unary{Tensor}, N}(
    __func::Function,
    row::Integer,
    column::Integer,
    data,
)::Nothing where {N, Tensor <: AbstractTensor}
    for i::typeof(column) in 0:(capacity(Val(Tensor), Val(N)) - 1)
        @inbounds data[row, column + i] = __func(data[row, column + i])
    end
    return nothing
end
