#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/13 21:39:50
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module DataIO

using JLD2
using CSV
using DataFrames
using CodecZstd
using OrderedCollections
using JSON
using YAML
using KernelAbstractions

using EtherParallelParticles.Utility
using EtherParallelParticles.Environment
using EtherParallelParticles.Class

include("Writer/Writer.jl")
include("Config/Config.jl")

end # module DataIO
