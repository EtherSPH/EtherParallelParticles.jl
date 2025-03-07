#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/03 20:56:32
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * getfield and getproperty are allowed on device!
 =#

# for NamedTuple

include("../../oneapi_head.jl")

@kernel function device_f!(x, nt::NamedTuple)
    I = @index(Global)
    x[I, 1] += getfield(nt, :a) # allowed
    x[I, 2] += getproperty(nt, :b) # allowed
end

a = zeros(Float32, 3, 2) |> CT
nt = (a = 1.0f0, b = 2.0f0)
device_f!(Backend, 256)(a, nt, ndrange = (3,))
KernelAbstractions.synchronize(Backend)
@info a
