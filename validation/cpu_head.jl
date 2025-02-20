#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/22 18:14:09
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using KernelAbstractions
using Random

const IT = Int32
const FT = Float32
const CT = Array
const Backend = KernelAbstractions.CPU()
