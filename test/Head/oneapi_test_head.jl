#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/10 17:56:09
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using EtherParallelParticles
using KernelAbstractions
import Pkg
Pkg.add("oneAPI")
using oneAPI

IT = Int32
FT = Float32
CT = oneArray
Backend = oneAPI.oneAPIBackend()
parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
