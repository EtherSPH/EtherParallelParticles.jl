#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/27 20:40:59
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description: # * works!
 =#

include("../../oneapi_head.jl")

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

@generated function pow(x, ::Val{N}) where {N}
    if N == 0
        return :(1)
    elseif N == 1
        return :(x)
    elseif N == 2
        return :(x * x)
    elseif N == 3
        return :(x * x * x)
    elseif N == 4
        return :(temp = x * x;
        temp * temp)
    elseif N == 5
        return :(temp = x * x;
        temp * temp * x)
    elseif N == 6
        return :(temp = x * x * x;
        temp * temp)
    else
        expr = :(x)
        for _ in 2:N
            expr = :($expr * x)
        end
        return expr
    end
end

@kernel function device_pow!(::Type{D}, x) where {N, D <: AbstractDimension{N}}
    I = @index(Global)
    @inbounds x[I] = pow(x[I], Val(N + 1))
end

x = Float32[1, 2, 3] |> CT
device_pow!(Backend, 3)(Dimension2D, x, ndrange = (3,))
KernelAbstractions.synchronize(Backend)
@info x
