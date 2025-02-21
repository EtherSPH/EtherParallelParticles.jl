#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/10 17:41:04
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using EtherParallelParticles
using KernelAbstractions
import Pkg
Pkg.add("CUDA")
using CUDA

IT = Int32
FT = Float32
CT = CuArray
Backend = CUDA.CUDABackend()
parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
