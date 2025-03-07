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
using EtherParallelParticles.Algorithm
using EtherParallelParticles.Math
using EtherParallelParticles.SPH.Macro

include("UpdateState.jl")
include("Continuity.jl")

end # module Library
