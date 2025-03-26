#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/22 21:33:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

# ! `action!` function should be decorated with `@inline` macro
# ! otherwise a `gpu malloc error` may occur

"""
# ! which is important is that on device,
# ! all variables are obtained from `offset`.
# ! thus count from 0 is more natural which is against the julia convention
"""
@inline function J(
    I::Integer,
    NI::Integer,
    IP,
    PM::NamedTuple, # must have field named `nIndex`
)::eltype(IP)
    @inbounds return IP[I, PM.nIndex + NI]
end

@inline function nullselfaction!(
    ::Type{Dimension},
    I::Integer,
    IP,
    FP,
    PM::NamedTuple,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    return nothing
end

@inline function nullinteraction!(
    ::Type{Dimension},
    I::Integer,
    NI::Integer,
    IP,
    FP,
    PM::NamedTuple,
)::Nothing where {N, Dimension <: AbstractDimension{N}}
    return nothing
end

# * ==================== selfaction! ==================== * #

@kernel function device_selfaction!(
    dimension::Type{Dimension},
    @Const(ps_is_alive),
    IP,
    FP,
    PM::NamedTuple,
    action!::Function,
) where {N, Dimension <: AbstractDimension{N}}
    I::eltype(IP) = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        action!(dimension, I, IP, FP, PM)
    end
end

@inline function selfaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    static_selfaction!(particle_system, action!; n_threads = n_threads) # default `static_selfaction!`
    return nothing
end

@inline function static_selfaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    kernel_selfaction! = device_selfaction!(Backend, n_threads, (Int64(Class.get_n_particles(particle_system)),))
    kernel_selfaction!(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_properties_,
        particle_system.base_.float_properties_,
        particle_system.parameters_,
        action!,
        ndrange = (Class.get_n_particles(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function dynamic_selfaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    device_selfaction!(Backend, n_threads)(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_properties_,
        particle_system.base_.float_properties_,
        particle_system.parameters_,
        action!,
        ndrange = (Class.get_n_particles(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
end

# * ==================== interaction! ==================== * #

@kernel function device_interaction!(
    dimension::Type{Dimension},
    @Const(ps_is_alive),
    IP,
    FP,
    PM::NamedTuple,
    index_nCount::IT,
    action!::Function,
) where {IT <: Integer, N, Dimension <: AbstractDimension{N}}
    I::eltype(IP) = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        NI::IT = IT(0)
        @inbounds while NI < IP[I, index_nCount]
            action!(dimension, I, NI, IP, FP, PM)
            NI += IT(1)
        end
    end
end

@inline function interaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    static_interaction!(particle_system, action!; n_threads = n_threads) # default `static_interaction!`
    return nothing
end

@inline function static_interaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    kernel_interaction! = device_interaction!(Backend, n_threads, (Int64(Class.get_n_particles(particle_system)),))
    kernel_interaction!(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_properties_,
        particle_system.base_.float_properties_,
        particle_system.parameters_,
        particle_system.basic_index_.nCount,
        action!,
        ndrange = (Class.get_n_particles(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function dynamic_interaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    device_interaction!(Backend, n_threads)(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_properties_,
        particle_system.base_.float_properties_,
        particle_system.parameters_,
        particle_system.basic_index_.nCount,
        action!,
        ndrange = (Class.get_n_particles(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
end
