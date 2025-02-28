#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/25 17:39:04
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

# # * ==================== dot ==================== * #

struct dot{Pattern <: AbstractPattern, N} <: AbstractOperator end

@inline function dot{Unary{__Vector}, N}(
    row_1::Integer,
    column_1::Integer,
    row_2::Integer,
    column_2::Integer,
    data,
)::eltype(data) where {N}
    result::eltype(data) = 0.0
    for i::typeof(column_1) in 0:(N - 1)
        @inbounds result += data[row_1, column_1 + i] * data[row_2, column_2 + i]
    end
    return result
end

@inline function dot{Unary{__Vector}, N}(
    row::Integer,
    column_1::Integer,
    column_2::Integer,
    data,
)::eltype(data) where {N}
    result::eltype(data) = 0.0
    for i::typeof(column_1) in 0:(N - 1)
        @inbounds result += data[row, column_1 + i] * data[row, column_2 + i]
    end
    return result
end

@inline function dot{Unary{__Vector}, N}(row::Integer, column::Integer, data)::eltype(data) where {N}
    result::eltype(data) = 0.0
    for i::typeof(column) in 0:(N - 1)
        @inbounds result += data[row, column + i] * data[row, column + i]
    end
    return result
end
