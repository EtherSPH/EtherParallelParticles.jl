#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/07 17:42:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Mean

@inline @fastmath function arithmetic(a::Real, b::Real)::typeof(a)
    return 0.5 * (a + b)
end

@inline @fastmath function invarithmetic(a::Real, b::Real)::typeof(a)
    return 2 / (a + b)
end

@inline @fastmath function geometric(a::Real, b::Real)::typeof(a)
    return sqrt(a * b)
end

@inline @fastmath function invgeometric(a::Real, b::Real)::typeof(a)
    return 1 / sqrt(a * b)
end

@inline @fastmath function harmonic(a::Real, b::Real)::typeof(a)
    return 2 * a * b / (a + b)
end

@inline @fastmath function invharmonic(a::Real, b::Real)::typeof(a)
    return (a + b) / (2 * a * b)
end

@inline @fastmath function quadratic(a::Real, b::Real)::typeof(a)
    return sqrt(a * a + b * b)
end

@inline @fastmath function invquadratic(a::Real, b::Real)::typeof(a)
    return 1 / sqrt(a * a + b * b)
end

end # module Mean
