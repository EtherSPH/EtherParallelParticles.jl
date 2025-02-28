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
# TODO: add `AMDGPU.jl` & `Metal.jl` test, literally it should work as well
const USING_CUDA, USING_ROC, USING_ONEAPI, USING_METAL = false, false, true, false

@testset "EtherParallelParticles" begin
    include("Utility/UtilityTest.jl")
    include("Environment/EnvironmentTest.jl")
    include("Class/ClassTest.jl")
    include("Algorithm/AlgorithmTest.jl")
end
