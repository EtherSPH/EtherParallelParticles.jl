#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/24 17:17:31
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function offset(start::Integer, off_set::Integer)::typeof(start)
    return start + off_set - 1
end

@inline function offset(start::Integer, off_set::Integer, step::Integer)::typeof(start)
    return start + (off_set - 1) * step
end

@inline function getItem(row::Integer, column::Integer, data)::eltype(data)
    @inbounds return data[row, column]
end

@inline function setItem!(value::Real, row::Integer, column::Integer, data)::Nothing
    @inbounds data[row, column] = eltype(data)(value)
    return nothing
end
