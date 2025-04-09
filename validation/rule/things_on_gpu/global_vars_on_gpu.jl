#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/09 22:02:13
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # ! global vars on GPU is not allowed.
 =#

include("../../oneapi_head.jl")

t = 1.0f0

@kernel function device_vadd!(a)
    global t
    I = @index(Global)
    a[I] += t
end

function host_vadd!(a)
    device_vadd!(Backend, 2)(a, ndrange = (2,))
    KernelAbstractions.synchronize(Backend)
end

a = zeros(FT, 2) |> CT

@info "before:"
@info "a = $(a)"
host_vadd!(a)
@info "after vadd:"
@info "a = $(a)"
