#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/04 17:58:57
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * this works!
 =#

module Macro

macro Index(symbol)
    return esc(:(getfield(PM, $symbol)))
end

macro Value(symbol)
    return esc(:(FP[I, getfield(PM, $symbol)]))
end

macro Value(symbol, i)
    return esc(:(FP[I, getfield(PM, $symbol) + $(j - 1)]))
end

macro Value(symbol, i, j)
    return esc(:(FP[I, getfield(PM, $symbol) + $(i - 1) + dimension(Dimension) * $(j - 1)]))
end

for name in ("Mass", "Volume", "Density", "Pressure")
    eval(Meta.parse("""
                    macro $name()
                        return esc(:(getfield(PM, $name)))
                    end
                    """))
    eval(Meta.parse("""
                    macro $name(I)
                        return esc(:(FP[\$I, PM.$name]))
                    end
                    """))
end

export Index, Value, Mass, Volume, Density, Pressure

end

include("../../oneapi_head.jl")

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

@inline function dimension(::Type{Dimension}) where {N, Dimension <: AbstractDimension{N}}
    return N
end

import .Macro as M

@inline function volume!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds M.@Volume(I) = M.@Mass(I) / M.@Density(I)
    return nothing
end

@inline function det(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    M.@Value(:StressMat, 1)
    return nothing
end

@kernel function device_f!(::Type{Dimension}, IP, FP, PM::NamedTuple) where {N, Dimension <: AbstractDimension{N}}
    I = @index(Global)
    volume!(Dimension, I, IP, FP, PM)
    det(Dimension, I, IP, FP, PM)
end

a = IT[
    1 2 3 4
    4 5 6 7
] |> CT
b = FT[
    1 2 3 4
    4 5 6 7
] |> CT
pm = (Mass = 1, Volume = 2, Density = 3, Pressure = 4, StressMat = 1) |> NamedTuple
device_f!(Backend, 256)(Dimension2D, a, b, pm, ndrange = (size(a, 1)))
KernelAbstractions.synchronize(Backend)
@info b
