#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 19:52:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Environment

using KernelAbstractions

include("Dimension.jl")
export AbstractDimension
export Dimension1D, Dimension2D, Dimension3D

include("Parallel.jl")
export AbstractParallel

include("Backend.jl")

end # module Environment
