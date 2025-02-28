#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 17:10:25
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

struct Gaussian{IT <: Integer, FT <: AbstractFloat, N} <: AbstractKernel{IT, FT, N} end

@inline radiusRatio(::Gaussian{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N} = IT(3)
@inline sigma(::Gaussian{IT, FT, 1}) where {IT <: Integer, FT <: AbstractFloat} = FT(1 / sqrt(pi))
@inline sigma(::Gaussian{IT, FT, 2}) where {IT <: Integer, FT <: AbstractFloat} = FT(1 / pi)
@inline sigma(::Gaussian{IT, FT, 3}) where {IT <: Integer, FT <: AbstractFloat} = FT(1 / sqrt(pi^3))

@inline @fastmath function _value(
    r::Real,
    h_inv::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * h_inv
    if q < FT(3.0)
        return sigma(kernel) * Math.power(h_inv, Val(N)) * exp(-q * q)
    else
        return FT(0.0)
    end
end

@inline @fastmath function value(
    r::Real,
    h::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value(r, 1 / h, kernel)
end

@inline @fastmath function _gradient(
    r::Real,
    h_inv::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * h_inv
    if q < FT(3.0)
        return -2 * sigma(kernel) * Math.power(h_inv, Val(N + 1)) * q * exp(-q * q)
    else
        return FT(0.0)
    end
end

@inline @fastmath function gradient(
    r::Real,
    h::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _gradient(r, 1 / h, kernel)
end
