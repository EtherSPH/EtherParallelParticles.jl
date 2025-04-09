#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/08 22:20:16
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

const kNameToContainer = Dict(
    "cpu" => "Array",
    "cuda" => "CUDA.CuArray",
    "rocm" => "AMDGPU.ROCArray",
    "oneapi" => "oneAPI.oneArray",
    "metal" => "Metal.MtlArray",
)

const kNameToBackend = Dict(
    "cpu" => "KernelAbstractions.CPU()",
    "cuda" => "CUDA.CUDABackend()",
    "rocm" => "AMDGPU.ROCBackend()",
    "oneapi" => "oneAPI.oneAPIBackend()",
    "metal" => "Metal.MetalBackend()",
)

const kContainerToName =
    Dict("Array" => "cpu", "CuArray" => "cuda", "ROCArray" => "rocm", "oneArray" => "oneapi", "MtlArray" => "metal")

const kNameToColor = Dict("cpu" => :magenta, "cuda" => :green, "rocm" => :red, "oneapi" => :blue, "metal" => :cyan)
