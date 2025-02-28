#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/24 19:37:40
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using EtherParallelParticles
using KernelAbstractions
import Pkg
Pkg.add("AMDGPU")
using AMDGPU

IT = Int32
FT = Float32
CT = AMDGPU.ROCArray
Backend = AMDGPU.ROCBackend()
parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
