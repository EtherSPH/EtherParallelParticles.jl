#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/27 17:01:14
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

struct CubicSpline{IT <: Integer, FT <: AbstractFloat, N} <: AbstractKernel{IT, FT, N} end

@inline radiusRatio(::CubicSpline{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N} = IT(2)
@inline sigma(::CubicSpline{IT, FT, 1}) where {IT <: Integer, FT <: AbstractFloat} = FT(2.0 / 3.0)
@inline sigma(::CubicSpline{IT, FT, 2}) where {IT <: Integer, FT <: AbstractFloat} = FT(10.0 / 7.0 / pi)
@inline sigma(::CubicSpline{IT, FT, 3}) where {IT <: Integer, FT <: AbstractFloat} = FT(1.0 / pi)

@inline @fastmath function _value(
    r::Real,
    hinv::Real,
    kernel::CubicSpline{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * hinv
    if q < FT(1.0)
        return sigma(kernel) * Math.power(hinv, Val(N)) * (3 * q * q * (q - 2) + 4) * FT(0.25)
    elseif q < FT(2.0)
        to_2::FT = 2 - q
        return sigma(kernel) * Math.power(hinv, Val(N)) * Math.power(to_2, Val(3)) * FT(0.25)
    else
        return FT(0.0)
    end
end

@inline @fastmath function value(
    r::Real,
    h::Real,
    kernel::CubicSpline{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value(r, 1 / h, kernel)
end

@inline @fastmath function _gradient(
    r::Real,
    hinv::Real,
    kernel::CubicSpline{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * hinv
    if q < FT(1.0)
        return sigma(kernel) * Math.power(hinv, Val(N + 1)) * q * (3 * q - 4) * FT(0.75)
    elseif q < FT(2.0)
        to_2::FT = 2 - q
        return -sigma(kernel) * Math.power(hinv, Val(N + 1)) * to_2 * to_2 * FT(0.75)
    else
        return FT(0.0)
    end
end

@inline @fastmath function gradient(
    r::Real,
    h::Real,
    kernel::CubicSpline{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _gradient(r, 1 / h, kernel)
end
