#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/24 15:33:04
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Math

abstract type AbstractTensor{Order} end
struct __Scalar <: AbstractTensor{0} end
struct __Vector <: AbstractTensor{1} end
struct __Matrix <: AbstractTensor{2} end
const Sca = __Scalar
const Vec = __Vector
const Mat = __Matrix

@inline capacity(::Val{Tensor}, ::Val{N}) where {Order, Tensor <: AbstractTensor{Order}, N} = N^Order

abstract type AbstractPattern end
struct Unary{Tensor <: AbstractTensor} <: AbstractPattern end
struct Binary{Tensor1 <: AbstractTensor, Tensor2 <: AbstractTensor} <: AbstractPattern end

abstract type AbstractOperator <: Function end

include("OptimizedFunction.jl")
include("Mean.jl")
include("Index.jl")
include("Basic.jl")
include("Broadcast.jl")
include("LinearAlgebra.jl")

export __Scalar, __Vector, __Matrix, Sca, Vec, Mat
export Unary, Binary
export capacity

end # module Math

using EtherParallelParticles.Math: Sca, Vec, Mat
using EtherParallelParticles.Math: Unary, Binary
using EtherParallelParticles.Math: capacity

export Sca, Vec, Mat
export Unary, Binary
export capacity
