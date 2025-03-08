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
using EtherParallelParticles.SPH.Macro
using EtherParallelParticles.SPH.Kernel

include("UpdateState.jl")
include("Kernel.jl")
include("Continuity.jl")
include("Pressure.jl")

end # module Library
