#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 20:30:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Library

using EtherParallelParticles.Environment
using EtherParallelParticles.Math
using EtherParallelParticles.SPH.Kernel
using EtherParallelParticles.SPH.Macro

@inline function avoidzero(r::Real, h::Real)::typeof(r)
    return r * r + typeof(r)(0.01) * h * h
end

@inline function vdotx(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
)::eltype(FP) where {N, Dimension <: AbstractDimension{N}}
    v_dot_x::@float() = @float 0.0
    for i::@int() in 0:(N - 1)
        @inbounds v_dot_x += @rvec(NI, i) * (@u(@i, i) - @u(@j, i))
    end
    return v_dot_x
end

include("State.jl")
include("Kernel.jl")
include("Motion.jl")
include("Continuity.jl")
include("Pressure.jl")
include("Viscosity.jl")
include("Filter.jl")

end # module Library
