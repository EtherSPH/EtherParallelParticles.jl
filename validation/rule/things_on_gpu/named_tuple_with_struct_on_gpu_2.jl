#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/24 17:14:26
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * changes on NamedTuple will not cause recompilation of the kernel
 =#

include("../../oneapi_head.jl")

using Accessors

struct LinearTransform{FT <: AbstractFloat}
    k_::FT
    b_::FT
end

@inline function (lt::LinearTransform)(x::FT)::FT where {FT <: AbstractFloat}
    return lt.k_ * x + lt.b_
end

@inline function lt(lt::LinearTransform{FT}, x::FT)::FT where {FT <: AbstractFloat}
    return lt.k_ * x + lt.b_
end

params = (a = 1.0f0, b = IT(2), lt = LinearTransform(2.0f0, 1.0f0))

@kernel function device_op!(a, params::NamedTuple)
    I = @index(Global)
    a[I] += params.a # 1
    a[I] *= params.b # 2
    a[I] = params.lt(a[I]) # 5
    a[I] = lt(params.lt, a[I]) # 11
end

function host_op!(a, params::NamedTuple)
    device_op!(Backend, 256)(a, params, ndrange = (length(a),))
    KernelAbstractions.synchronize(Backend)
end

a = zeros(FT, 2) |> CT

@info "before:"
@info "a = $(a)"
host_op!(a, params)
@info "after op:"
@info "a = $(a)"
@reset params.a = 0.0f0
@reset params.b = IT(0)
@reset params.lt.k_ = 0.0f0
@reset params.lt.b_ = 0.0f0
host_op!(a, params)
@info "after reset:"
@info "a = $(a)"
