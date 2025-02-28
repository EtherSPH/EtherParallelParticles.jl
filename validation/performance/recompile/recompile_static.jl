#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/24 21:53:24
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

include("../../oneapi_head.jl")

@kernel function device_add!(a, b)
    I = @index(Global)
    a[I] += b[I]
end

@inline function host_add!(a, b)
    kernel_add! = device_add!(Backend, 256, (length(a),))
    kernel_add!(a, b, ndrange = (length(a),))
    KernelAbstractions.synchronize(Backend)
end

a1 = randn(FT, 1024) |> CT
b1 = randn(FT, 1024) |> CT
a2 = randn(FT, 512) |> CT
b2 = randn(FT, 512) |> CT

@time host_add!(a1, b1)
@time host_add!(a1, b1)
@time host_add!(a2, b2)
@time host_add!(a2, b2)

# causing recompilation as the `length(a)` varies
# 13.093194 seconds (14.71 M allocations: 745.231 MiB, 2.09% gc time, 99.11% compilation time: 2% of which was recompilation)
# 0.006858 seconds (57 allocations: 2.047 KiB)
# 0.684284 seconds (188.83 k allocations: 9.261 MiB, 66.84% compilation time)
# 0.004691 seconds (57 allocations: 2.047 KiB)
