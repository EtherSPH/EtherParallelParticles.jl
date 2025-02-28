#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/24 17:43:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * ops on gpu is allowed
 =#

include("../../oneapi_head.jl")

@inline function apply_op(op::Function, x, y)
    return op(x, y)
end

@kernel function some_op(x)
    I = @index(Global)
    @inbounds x[I, 1] = apply_op(+, x[I, 2], x[I, 3])
end

a = rand(Float32, 4, 3) |> CT
@info a
some_op(Backend, 4)(a, ndrange = (4,))
KernelAbstractions.synchronize(Backend)
@info a
