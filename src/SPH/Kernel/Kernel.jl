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

export AbstractKernel

include("CubicSpline.jl")
include("Gaussian.jl")
include("WendlandC2.jl")
include("WendlandC4.jl")

@inline @fastmath function _value0(
    h_inv::Real,
    kernel::AbstractKernel{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return sigma(kernel) * Math.power(h_inv, Val(N))
end

@inline @fastmath function value0(
    h::Real,
    kernel::AbstractKernel{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value0(1 / h, kernel)
end

const W = value
const DW = gradient
const _W = _value
const _DW = _gradient
const _W0 = _value0

end # module Kernel
