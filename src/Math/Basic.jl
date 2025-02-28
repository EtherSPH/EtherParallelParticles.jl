#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/25 17:31:47
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

struct fill!{Tensor <: AbstractTensor, N} <: AbstractOperator end
struct reset!{Tensor <: AbstractTensor, N} <: AbstractOperator end

@inline function fill!(value::Real, n::Integer, row::Integer, column::Integer, data)::Nothing
    for i::typeof(row) in 0:(n - 1)
        @inbounds data[row, column + i] = eltype(data)(value)
    end
    return nothing
end

@inline function reset!(n::Integer, row::Integer, column::Integer, data)::Nothing
    for i::typeof(row) in 0:(n - 1)
        @inbounds data[row, column + i] = zero(eltype(data))
    end
    return nothing
end

@inline function fill!{Tensor, N}(
    value::Real,
    row::Integer,
    column::Integer,
    data,
)::Nothing where {N, Tensor <: AbstractTensor}
    fill!(value, capacity(Val(Tensor), Val(N)), row, column, data)
    return nothing
end

@inline function reset!{Tensor, N}(row::Integer, column::Integer, data)::Nothing where {N, Tensor <: AbstractTensor}
    reset!(capacity(Val(Tensor), Val(N)), row, column, data)
    return nothing
end
