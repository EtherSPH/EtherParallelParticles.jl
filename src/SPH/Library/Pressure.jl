#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/07 21:56:57
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline function pressureCorrection(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
    kernel::AbstractKernel;
    factor::Real = 1,
)::eltype(FP) where {N, Dimension <: AbstractDimension{N}}
    return @float(0.01 * factor) * @w(@ij) / Kernel._value(Math.Mean.arithmetic(@gap(@i), @gap(@j)), @hinv(@ij), kernel)
end

@inline function classicPressure!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    coefficient::Real = 0, # 0.01 w(r) / w(Δx)
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    p_rho_2::@float() = @p(@i) / (@rho(@i) * @rho(@i)) + @p(@j) / (@rho(@j) * @rho(@j))
    p_rho_2 += abs(p_rho_2) * @float(coefficient)
    p_rho_2 *= -@mass(@j) * @dw(@ij) / @r(@ij)
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @du(@i, i) += p_rho_2 * @rvec(@ij, i)
    end
    return nothing
end

@inline function balancedPressure!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    coefficient::Real = 0, # 0.01 w(r) / w(Δx)
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    p_rho_2::@float() = (@p(@i) + @p(@j)) / (@rho(@i) * @rho(@j))
    p_rho_2 += abs(p_rho_2) * @float(coefficient)
    p_rho_2 *= -@mass(@j) * @dw(@ij) / @r(@ij)
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @du(@i, i) += p_rho_2 * @rvec(@ij, i)
    end
    return nothing
end

@inline function densityWeightedPressure!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    coefficient::Real = 0, # 0.01 w(r) / w(Δx)
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    p_rho_2::@float() = 2 * (@p(@i) * @rho(@i) + @p(@j) * @rho(@j)) / ((@rho(@i) + @rho(@j)) * @rho(@i) * @rho(@j))
    p_rho_2 += abs(p_rho_2) * @float(coefficient)
    p_rho_2 *= -@mass(@j) * @dw(@ij) / @r(@ij)
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @du(@i, i) += p_rho_2 * @rvec(@ij, i)
    end
    return nothing
end

@inline function extrapolatePressure!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    p0::Real = 0, # background pressure
    gx::Real = 0,
    gy::Real = 0,
)::Nothing where {Dimension <: AbstractDimension{2}}
    g_dot_x::@float() = @float 0.0
    g_dot_x += @rvec(@ij, 0) * @float(gx)
    g_dot_x += @rvec(@ij, 1) * @float(gy)
    @inbounds @wv(@i) += @w(@ij) * @vol(@j)
    @inbounds @wv_p(@i) += @wv(@i) * (max(@p(@j), @float(p0)) + max(@rho(@j) * g_dot_x, @float(p0)))
    return nothing
end

@inline function extrapolatePressure!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple;
    p0::Real = 0, # background pressure
    gx::Real = 0,
    gy::Real = 0,
    gz::Real = 0,
)::Nothing where {Dimension <: AbstractDimension{3}}
    g_dot_x::@float() = @float 0.0
    g_dot_x += @rvec(@ij, 0) * @float(gx)
    g_dot_x += @rvec(@ij, 1) * @float(gy)
    g_dot_x += @rvec(@ij, 2) * @float(gz)
    wv::@float() = @w(@ij) * @vol(@j)
    @inbounds @wv(@i) += wv
    @inbounds @wv_p(@i) += wv * (max(@p(@j), @float(p0)) + max(@rho(@j) * g_dot_x, @float(p0)))
    return nothing
end

@inline function extrapolatePressure!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple;
    p0::Real = 0, # background pressure
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    @inbounds if @wv(@i) > @float(0.0)
        @inbounds @p(@i) = @wv_p(@i) / @wv(@i)
        @inbounds @wv(@i) = @float 0.0
        @inbounds @wv_p(@i) = @float 0.0
        return nothing
    else
        @inbounds @p(@i) = @float p0
        return nothing
    end
    return nothing
end
