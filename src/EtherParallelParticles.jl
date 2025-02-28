#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 19:40:26
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module EtherParallelParticles

include("Utility/Utility.jl")
include("Environment/Environment.jl")
include("Class/Class.jl")
include("Algorithm/Algorithm.jl")
include("Math/Math.jl")
include("SPH/SPH.jl")

export Utility, Environment, Class, Algorithm, Math, SPH

end # module EtherParallelParticles
