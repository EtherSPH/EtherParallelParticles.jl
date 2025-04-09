#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/07 17:08:07
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Geometry

using EtherParallelParticles.Environment
using EtherParallelParticles.Class

abstract type AbstractGeometry{N} end

include("Geometry2D.jl")
include("Geometry3D.jl")

end # module Geometry
