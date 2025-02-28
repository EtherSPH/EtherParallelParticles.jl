#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/11 14:46:03
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Class

using KernelAbstractions

using EtherParallelParticles.Environment

const kDefaultThreadNumber = 256
const kDefaultMaxNeighbourNumber = 50

include("Domain/AbstractDomain.jl")
export AbstractDomain
export Domain2D

include("ParticleSystem/ParticleSystem.jl")
export AbstractParticleSystem
export ParticleSystem

include("NeighbourSystem/NeighbourSystem.jl")
export AbstractNeighbourSystem
export NeighbourSystem
export AbstractPeriodicBoundaryPolicy
export NonePeriodicBoundaryPolicy, HavePeriodicBoundaryPolicy
export PeriodicBoundaryPolicy2D, PeriodicBoundaryPolicy3D

end # module Class

using EtherParallelParticles.Class: Domain2D
using EtherParallelParticles.Class: ParticleSystem
using EtherParallelParticles.Class: NeighbourSystem
using EtherParallelParticles.Class: PeriodicBoundaryPolicy2D
using EtherParallelParticles.Class: PeriodicBoundaryPolicy3D

export Domain2D
export ParticleSystem
export NeighbourSystem
export PeriodicBoundaryPolicy2D
export PeriodicBoundaryPolicy3D
