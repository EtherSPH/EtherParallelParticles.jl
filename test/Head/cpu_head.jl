#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/10 17:28:50
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using EtherParallelParticles
using KernelAbstractions

const IT = Int32
const FT = Float32
const CT = Array
const Backend = KernelAbstractions.CPU()
const parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
