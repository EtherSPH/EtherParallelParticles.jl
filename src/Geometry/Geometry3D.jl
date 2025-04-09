#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/07 17:15:17
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

struct Cuboid{T <: Real} <: AbstractGeometry{3}
    first_x_::T
    first_y_::T
    first_z_::T
    last_x_::T
    last_y_::T
    last_z_::T
    span_x_::T
    span_y_::T
    span_z_::T
end

function Cuboid{T}(x1::Real, y1::Real, z1::Real, x2::Real, y2::Real, z2::Real) where {T <: Real}
    @assert x1 < x2 "x1 must be less than x2"
    @assert y1 < y2 "y1 must be less than y2"
    @assert z1 < z2 "z1 must be less than z2"
    return Cuboid{T}(T(x1), T(y1), T(z1), T(x2), T(y2), T(z2), T(x2 - x1), T(y2 - y1), T(z2 - z1))
end

function Cuboid(x1::Real, y1::Real, z1::Real, x2::Real, y2::Real, z2::Real)
    return Cuboid{typeof(x1)}(x1, y1, z1, x2, y2, z2)
end

@inline function get_first_x(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.first_x_
end

@inline function get_first_y(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.first_y_
end

@inline function get_first_z(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.first_z_
end

@inline get_x(cuboid::Cuboid) = get_first_x(cuboid)
@inline get_y(cuboid::Cuboid) = get_first_y(cuboid)
@inline get_z(cuboid::Cuboid) = get_first_z(cuboid)

@inline function get_last_x(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.last_x_
end

@inline function get_last_y(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.last_y_
end

@inline function get_last_z(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.last_z_
end

@inline function get_span_x(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.span_x_
end

@inline function get_span_y(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.span_y_
end

@inline function get_span_z(cuboid::Cuboid{T})::T where {T <: Real}
    return cuboid.span_z_
end

@inline function count(gap::Real, cuboid::Cuboid{T}) where {T <: Real}
    n_x = Int(round(get_span_x(cuboid) / gap))
    n_y = Int(round(get_span_y(cuboid) / gap))
    n_z = Int(round(get_span_z(cuboid) / gap))
    return n_x * n_y * n_z
end

@inline function inside(x::Real, y::Real, z::Real, cuboid::Cuboid{T}) where {T <: Real}
    return x >= get_first_x(cuboid) &&
           x <= get_last_x(cuboid) &&
           y >= get_first_y(cuboid) &&
           y <= get_last_y(cuboid) &&
           z >= get_first_z(cuboid) &&
           z <= get_last_z(cuboid)
end

@inline function create(gap::Real, cuboid::Cuboid{T}) where {T <: Real}
    n_x = Int(round(get_span_x(cuboid) / gap))
    n_y = Int(round(get_span_y(cuboid) / gap))
    n_z = Int(round(get_span_z(cuboid) / gap))
    # TODO
end
