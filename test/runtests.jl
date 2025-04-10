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
using EtherParallelParticles.SPH.Macro
using KernelAbstractions

# support for `cpu`, `cuda`, `rocm`, `oneapi`, `metal`
const DEVICE = "oneapi"
include("Head/$(DEVICE)_head.jl")
@info "test on backend: $DEVICE"

@testset "EtherParallelParticles" begin
    include("Utility/UtilityTest.jl")
    include("Environment/EnvironmentTest.jl")
    include("Class/ClassTest.jl")
    include("Algorithm/AlgorithmTest.jl")
    include("Geometry/GeometryTest.jl")
    include("Math/MathTest.jl")
    include("SPH/SPHTest.jl")
end
