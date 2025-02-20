#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/14 17:01:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using KernelAbstractions
using CUDA
using Random

const IT = Int32
const FT = Float32
const CT = CUDA.CuArray
const Backend = CUDA.CUDABackend()
