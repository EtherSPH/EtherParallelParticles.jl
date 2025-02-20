#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 17:00:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using KernelAbstractions
using oneAPI
using Random

const IT = Int32
const FT = Float32
const CT = oneAPI.oneArray
const Backend = oneAPI.oneAPIBackend()
