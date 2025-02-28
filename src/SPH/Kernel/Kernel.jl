#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/27 16:47:43
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Kernel

using EtherParallelParticles.Math

abstract type AbstractKernel{IT <: Integer, FT <: AbstractFloat, N} end

include("CubicSpline.jl")
include("Gaussian.jl")
include("WendlandC2.jl")
include("WendlandC4.jl")

const W = value
const DW = gradient
const _W = _value
const _DW = _gradient

end # module Kernel
