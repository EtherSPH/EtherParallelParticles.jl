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

IT = Int32
FT = Float32
CT = Array
Backend = KernelAbstractions.CPU()
parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
