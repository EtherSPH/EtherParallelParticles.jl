#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 20:16:55
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using Test
using EtherParallelParticles
using KernelAbstractions

const USING_CPU = true
# it seems USING_CUDA and USING_ONEAPI may conflict with each other
# set only one of them to true
const USING_CUDA = false
const USING_ONEAPI = true

@testset "EtherParallelParticles" begin
    include("Utility/UtilityTest.jl")
    include("Environment/EnvironmentTest.jl")
    include("Class/ClassTest.jl")
    include("Algorithm/AlgorithmTest.jl")
end
