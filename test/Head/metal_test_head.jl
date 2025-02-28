#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/24 19:34:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using EtherParallelParticles
using KernelAbstractions
import Pkg
Pkg.add("Metal")
using Metal

IT = Int32
FT = Float32
CT = Metal.MtlArray
Backend = Metal.MetalBackend()
parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
