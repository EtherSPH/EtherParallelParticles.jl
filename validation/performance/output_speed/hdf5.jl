#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/25 14:02:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * hdf5 is a littel faster than jld2.
 =#

using HDF5
using BenchmarkTools

file = "drafts/output.hdf5"
if isdir("drafts")
    nothing
else
    mkdir("drafts")
end

const n_int = 100
const n_float = 500
const m = 100_000
const m_mask = m - 1

int = ones(Int32, m, n_int)
float = randn(Float32, m, n_float)
_mask = ones(Int32, m)
_mask[1:(end - 1)] .= 1
time_dict = Dict("a" => 0, "b" => 1, "c" => 2.0f0)

@info "warmup"
h5open(file, "w") do file
    for (key, value) in time_dict
        write(file, "params/$key", value)
    end
    mask = _mask .== 1
    write(file, "data/int", int[mask, :])
    write(file, "data/float", float[mask, :])
end

@info "benchmark"

@benchmark h5open(file, "w") do file
    for (key, value) in time_dict
        write(file, "params/$key", value)
    end
    mask = _mask .== 1
    write(file, "data/int", int[mask, :])
    write(file, "data/float", float[mask, :])
end
