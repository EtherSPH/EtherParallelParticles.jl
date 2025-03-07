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

const IT = Int32
const FT = Float32
const CT = CuArray
const Backend = CUDA.CUDABackend()
const parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
