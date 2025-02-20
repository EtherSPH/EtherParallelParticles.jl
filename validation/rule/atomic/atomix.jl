#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/18 22:25:35
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # ! important: `oneAPI.jl` currently only supports atomic operations for `Int32` types.
 =#

include("../../oneapi_head.jl")

using Atomix

const n = 10

a = Vector{IT}(1:n) |> CT
b = zeros(IT, 1) |> CT
c = zeros(IT, n) |> CT

@kernel function device_atomic_add!(a, b, c)
    I = @index(Global)
    c[I] = Atomix.@atomic b[1] += a[I]
end

function host_atomic_add!(a, b, c)
    device_atomic_add!(Backend, n)(a, b, c, ndrange = (n,))
    KernelAbstractions.synchronize(Backend)
end

host_atomic_add!(a, b, c)
@info "b = $(b)"
@info "c = $(c)"
