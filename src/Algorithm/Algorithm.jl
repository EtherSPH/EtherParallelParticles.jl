#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/18 22:03:18
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Algorithm

using KernelAbstractions
using Atomix

using EtherParallelParticles.Environment
using EtherParallelParticles.Class

const kDefaultThreadNumber = 256
const kDefaultMaxNeighbourNumber = 50

include("NeighbourSearch.jl")

end # module Algorithm
