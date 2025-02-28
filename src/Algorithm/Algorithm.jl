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
include("Action.jl")

end # module Algorithm

using EtherParallelParticles.Algorithm: search!, static_search!, dynamic_search!
using EtherParallelParticles.Algorithm: selfaction!, static_selfaction!, dynamic_selfaction!
using EtherParallelParticles.Algorithm: interaction!, static_interaction!, dynamic_interaction!

export search!, static_search!, dynamic_search!
export selfaction!, static_selfaction!, dynamic_selfaction!
export interaction!, static_interaction!, dynamic_interaction!
