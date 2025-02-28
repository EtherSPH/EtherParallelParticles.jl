#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 17:25:10
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

struct WendlandC2{IT <: Integer, FT <: AbstractFloat, N} <: AbstractKernel{IT, FT, N} end

@inline radiusRatio(::WendlandC2{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N} = IT(2)
@inline sigma(::WendlandC2{IT, FT, 1}) where {IT <: Integer, FT <: AbstractFloat} = FT(0.0)
@inline sigma(::WendlandC2{IT, FT, 2}) where {IT <: Integer, FT <: AbstractFloat} = FT(7.0 / 4.0 / pi)
@inline sigma(::WendlandC2{IT, FT, 3}) where {IT <: Integer, FT <: AbstractFloat} = FT(21.0 / 16.0 / pi)

@inline @fastmath function _value(
    r::Real,
    h_inv::Real,
    kernel::WendlandC2{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * h_inv
    if q < FT(2.0)
        return sigma(kernel) * Math.power(2 - q, Val(4)) * Math.power(h_inv, Val(N)) * (1 + 2 * q) * FT(0.0625)
    else
        return FT(0.0)
    end
end

@inline @fastmath function value(
    r::Real,
    h::Real,
    kernel::WendlandC2{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value(r, 1 / h, kernel)
end

@inline @fastmath function _gradient(
    r::Real,
    h_inv::Real,
    kernel::WendlandC2{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * h_inv
    if q < FT(2.0)
        return -sigma(kernel) * Math.power(h_inv, Val(N + 1)) * FT(0.625) * q * Math.power(2 - q, Val(3))
    else
        return FT(0.0)
    end
end

@inline @fastmath function gradient(
    r::Real,
    h::Real,
    kernel::WendlandC2{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _gradient(r, 1 / h, kernel)
end
