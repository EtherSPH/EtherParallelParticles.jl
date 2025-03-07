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

const IT = Int32
const FT = Float32
const CT = Metal.MtlArray
const Backend = Metal.MetalBackend()
const parallel = EtherParallelParticles.Environment.Parallel{IT, FT, CT, Backend}()
