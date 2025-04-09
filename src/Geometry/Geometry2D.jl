#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/07 17:15:06
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

# * ===================== Rectangle ===================== * #

struct Rectangle{T <: Real} <: AbstractGeometry{2}
    first_x_::T
    first_y_::T
    last_x_::T
    last_y_::T
    span_x_::T
    span_y_::T
end

function Rectangle{T}(x1::Real, y1::Real, x2::Real, y2::Real) where {T <: Real}
    @assert x1 < x2 "x1 must be less than x2"
    @assert y1 < y2 "y1 must be less than y2"
    return Rectangle{T}(T(x1), T(y1), T(x2), T(y2), T(x2 - x1), T(y2 - y1))
end

function Rectangle(x1::Real, y1::Real, x2::Real, y2::Real)
    return Rectangle{typeof(x1)}(x1, y1, x2, y2)
end

@inline function get_first_x(rectangle::Rectangle{T})::T where {T <: Real}
    return rectangle.first_x_
end

@inline function get_first_y(rectangle::Rectangle{T})::T where {T <: Real}
    return rectangle.first_y_
end

@inline get_x(rectangle::Rectangle) = get_first_x(rectangle)
@inline get_y(rectangle::Rectangle) = get_first_y(rectangle)

@inline function get_last_x(rectangle::Rectangle{T})::T where {T <: Real}
    return rectangle.last_x_
end

@inline function get_last_y(rectangle::Rectangle{T})::T where {T <: Real}
    return rectangle.last_y_
end

@inline function get_span_x(rectangle::Rectangle{T})::T where {T <: Real}
    return rectangle.span_x_
end

@inline function get_span_y(rectangle::Rectangle{T})::T where {T <: Real}
    return rectangle.span_y_
end

@inline function count(gap::Real, rectangle::Rectangle{T}) where {T <: Real}
    n_x = Int(round(get_span_x(rectangle) / gap))
    n_y = Int(round(get_span_y(rectangle) / gap))
    return n_x * n_y
end

@inline function inside(x::Real, y::Real, rectangle::Rectangle{T}) where {T <: Real}
    return x >= get_first_x(rectangle) &&
           x <= get_last_x(rectangle) &&
           y >= get_first_y(rectangle) &&
           y <= get_last_y(rectangle)
end

@inline function create(gap::Real, rectangle::Rectangle{T}; parallel::Bool = false) where {T <: Real}
    n_x = Int(round(get_span_x(rectangle) / gap))
    n_y = Int(round(get_span_y(rectangle) / gap))
    n = n_x * n_y
    dx = get_span_x(rectangle) / n_x
    dy = get_span_y(rectangle) / n_y
    volume = dx * dy
    _gap = sqrt(dx * dy)
    positions = zeros(T, n, 2)
    volumes = zeros(T, n)
    gaps = zeros(T, n)
    function single!(i)
        i_x = mod(i, n_x)
        i_y = div(i - 1, n_x) + 1
        x = get_first_x(rectangle) + (i_x - 0.5) * dx
        y = get_first_y(rectangle) + (i_y - 0.5) * dy
        positions[i, :] .= [x, y]
        volumes[i] = volume
        gaps[i] = _gap
    end
    if parallel == true
        Threads.@threads for i in 1:n
            single!(i)
        end
    else
        for i in 1:n
            single!(i)
        end
    end
    return positions, volumes, gaps
end

#* ===================== Ring ===================== * #

struct Ring{T <: Real} <: AbstractGeometry{2}
    center_x_::T
    center_y_::T
    inner_radius_::T
    outer_radius_::T
    radius_distance_::T
end

function Ring{T}(x::Real, y::Real, inner_radius::Real, outer_radius::Real) where {T <: Real}
    @assert inner_radius < outer_radius "inner_radius must be less than outer_radius"
    return Ring{T}(T(x), T(y), T(inner_radius), T(outer_radius), T(outer_radius - inner_radius))
end

function Ring(x::Real, y::Real, inner_radius::Real, outer_radius::Real)
    return Ring{typeof(x)}(x, y, inner_radius, outer_radius)
end

@inline function get_center_x(ring::Ring{T})::T where {T <: Real}
    return ring.center_x_
end

@inline function get_center_y(ring::Ring{T})::T where {T <: Real}
    return ring.center_y_
end

@inline get_x(ring::Ring) = get_center_x(ring)
@inline get_y(ring::Ring) = get_center_y(ring)

@inline function radius(x::Real, y::Real, ring::Ring{T}) where {T <: Real}
    dx = x - get_center_x(ring)
    dy = y - get_center_y(ring)
    return sqrt(dx * dx + dy * dy)
end

@inline function get_inner_radius(ring::Ring{T})::T where {T <: Real}
    return ring.inner_radius_
end

@inline function get_outer_radius(ring::Ring{T})::T where {T <: Real}
    return ring.outer_radius_
end

@inline function get_radius_distance(ring::Ring{T})::T where {T <: Real}
    return ring.radius_distance_
end

@inline function count(gap::Real, ring::Ring{T}) where {T <: Real}
    n_r = Int(round(get_radius_distance(ring) / gap))
    dr = get_radius_distance(ring) / n_r
    n_theta_s = zeros(Int, n_r)
    for i_r in 1:n_r
        r = get_inner_radius(ring) + (i_r - 0.5) * dr
        perimeter = 2 * pi * r
        n_theta_s[i_r] = Int(round(perimeter / gap))
    end
    return sum(n_theta_s)
end

@inline function inside(x::Real, y::Real, ring::Ring{T}) where {T <: Real}
    r = radius(x, y, ring)
    return r >= get_inner_radius(ring) && r <= get_outer_radius(ring)
end

@inline function create(gap::Real, ring::Ring{T}; parallel::Bool = false) where {T <: Real}
    n_r = Int(round(get_radius_distance(ring) / gap))
    dr = get_radius_distance(ring) / n_r
    n_theta_s = zeros(Int, n_r)
    for i_r in 1:n_r
        r = get_inner_radius(ring) + (i_r - 0.5) * dr
        perimeter = 2 * pi * r
        n_theta_s[i_r] = Int(round(perimeter / gap))
    end
    n = sum(n_theta_s)
    accumulated_n_theta_s = cumsum(n_theta_s)
    volumes = zeros(T, n)
    gaps = zeros(T, n)
    positions = zeros(T, n, 2)
    function single!(i)
        i_r = findfirst(x -> x >= i, accumulated_n_theta_s)
        i_theta = i - accumulated_n_theta_s[i_r]
        r = get_inner_radius(ring) + (i_r - 0.5) * dr
        n_theta = n_theta_s[i_r]
        dtheta = 2 * pi / n_theta
        theta = (i_theta - 0.5) * dtheta
        x = get_center_x(ring) + r * cos(theta)
        y = get_center_y(ring) + r * sin(theta)
        positions[i, :] .= [x, y]
        volumes[i] = dr * r * dtheta
        gaps[i] = sqrt(volumes[i])
    end
    if parallel == true
        Threads.@threads for i in 1:n
            single!(i)
        end
    else
        for i in 1:n
            single!(i)
        end
    end
    return positions, volumes, gaps
end

# * ===================== Circle ===================== * #

struct Circle{T <: Real} <: AbstractGeometry{2}
    center_x_::T
    center_y_::T
    radius_::T
end

function Circle{T}(x::Real, y::Real, radius::Real) where {T <: Real}
    @assert radius > 0 "radius must be greater than 0"
    return Circle{T}(T(x), T(y), T(radius))
end

function Circle(x::Real, y::Real, radius::Real)
    return Circle{typeof(x)}(x, y, radius)
end

@inline function get_center_x(circle::Circle{T})::T where {T <: Real}
    return circle.center_x_
end

@inline function get_center_y(circle::Circle{T})::T where {T <: Real}
    return circle.center_y_
end

@inline get_x(circle::Circle) = get_center_x(circle)
@inline get_y(circle::Circle) = get_center_y(circle)

@inline function get_radius(circle::Circle{T})::T where {T <: Real}
    return circle.radius_
end

@inline function radius(x::Real, y::Real, circle::Circle{T}) where {T <: Real}
    dx = x - get_center_x(circle)
    dy = y - get_center_y(circle)
    return sqrt(dx * dx + dy * dy)
end

@inline function count(gap::Real, circle::Circle{T}) where {T <: Real}
    inner_radius = 0.5 * gap
    outer_radius = get_radius(circle)
    n_r = Int(round((outer_radius - inner_radius) / gap))
    dr = (outer_radius - inner_radius) / n_r
    n_theta_s = zeros(Int, n_r)
    for i_r in 1:n_r
        r = inner_radius + (i_r - 0.5) * dr
        perimeter = 2 * pi * r
        n_theta_s[i_r] = Int(round(perimeter / gap))
    end
    return sum(n_theta_s) + 1
end

@inline function inside(x::Real, y::Real, circle::Circle{T}) where {T <: Real}
    r = radius(x, y, circle)
    return r <= get_radius(circle)
end

@inline function create(gap::Real, circle::Circle{T}; parallel::Bool = false) where {T <: Real}
    inner_radius = 0.5 * gap
    outer_radius = get_radius(circle)
    n_r = Int(round((outer_radius - inner_radius) / gap))
    dr = (outer_radius - inner_radius) / n_r
    n_theta_s = zeros(Int, n_r)
    for i_r in 1:n_r
        r = inner_radius + (i_r - 0.5) * dr
        perimeter = 2 * pi * r
        n_theta_s[i_r] = Int(round(perimeter / gap))
    end
    n = sum(n_theta_s) + 1
    accumulated_n_theta_s = cumsum(n_theta_s)
    volumes = zeros(T, n)
    gaps = zeros(T, n)
    positions = zeros(T, n, 2)
    function single!(i)
        if i == 1
            positions[i, :] .= [get_center_x(circle), get_center_y(circle)]
            volumes[i] = (0.5 * gap)^2 * pi
            gaps[i] = sqrt(volumes[i])
        else
            i_r = findfirst(x -> x >= i - 1, accumulated_n_theta_s)
            i_theta = i - 1 - accumulated_n_theta_s[i_r]
            r = inner_radius + (i_r - 0.5) * dr
            n_theta = n_theta_s[i_r]
            dtheta = 2 * pi / n_theta
            theta = (i_theta - 0.5) * dtheta
            x = get_center_x(circle) + r * cos(theta)
            y = get_center_y(circle) + r * sin(theta)
            positions[i, :] .= [x, y]
            volumes[i] = dr * r * dtheta
            gaps[i] = sqrt(volumes[i])
        end
    end
    if parallel == true
        Threads.@threads for i in 1:n
            single!(i)
        end
    else
        for i in 1:n
            single!(i)
        end
    end
    return positions, volumes, gaps
end
