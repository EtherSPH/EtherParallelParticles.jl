#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/28 17:39:13
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

struct WendlandC4{IT <: Integer, FT <: AbstractFloat, N} <: AbstractKernel{IT, FT, N} end

@inline radiusRatio(::WendlandC4{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N} = IT(2)
@inline sigma(::WendlandC4{IT, FT, 1}) where {IT <: Integer, FT <: AbstractFloat} = FT(5.0 / 8.0)
@inline sigma(::WendlandC4{IT, FT, 2}) where {IT <: Integer, FT <: AbstractFloat} = FT(9.0 / 4.0 / pi)
@inline sigma(::WendlandC4{IT, FT, 3}) where {IT <: Integer, FT <: AbstractFloat} = FT(495.0 / 256.0 / pi)

@inline @fastmath function _value(
    r::Real,
    h_inv::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * h_inv
    if q < FT(2.0)
        return sigma(kernel) *
               Math.power(h_inv, Val(N)) *
               Math.power(2 - q, Val(6)) *
               (q * (35 * q + 36) + 12) *
               FT(0.0013020833333333333)
    else
        return FT(0.0)
    end
end

@inline @fastmath function value(
    r::Real,
    h::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value(r, 1 / h, kernel)
end

@inline @fastmath function _gradient(
    r::Real,
    h_inv::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * h_inv
    if q < FT(2.0)
        return -sigma(kernel) *
               Math.power(h_inv, Val(N + 1)) *
               q *
               (FT(0.3645833333333333) * q + FT(0.14583333333333334)) *
               Math.power(2 - q, Val(5))
    else
        return FT(0.0)
    end
end

@inline @fastmath function gradient(
    r::Real,
    h::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _gradient(r, 1 / h, kernel)
end
