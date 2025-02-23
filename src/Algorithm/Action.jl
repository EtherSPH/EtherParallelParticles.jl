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

@inline function J(
    I::Integer,
    NI::Integer,
    ps_int_properties,
    parameters::NamedTuple, # must have field named `nIndex`
)::eltype(ps_int_properties)
    @inbounds return ps_int_properties[I, parameters.nIndex + NI - 1]
end

@kernel function device_selfaction!(
    dimension::Type{Dimension},
    @Const(ps_is_alive),
    ps_int_properties,
    ps_float_properties,
    parameters::NamedTuple,
    action!::Function,
) where {N, Dimension <: AbstractDimension{N}}
    I::eltype(ps_int_properties) = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        action!(dimension, I, ps_int_properties, ps_float_properties, parameters)
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
    device_selfaction!(Backend, n_threads)(
        Dimension,
        particle_system.device_base_.is_alive_,
        particle_system.device_base_.int_properties_,
        particle_system.device_base_.float_properties_,
        particle_system.parameters_,
        action!,
        ndrange = (Class.get_n_particles(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@kernel function device_interaction!(
    dimension::Type{Dimension},
    @Const(ps_is_alive),
    ps_int_properties,
    ps_float_properties,
    parameters::NamedTuple,
    index_nCount::IT,
    action!::Function,
) where {IT <: Integer, N, Dimension <: AbstractDimension{N}}
    I::eltype(ps_int_properties) = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        NI::IT = IT(1)
        @inbounds while NI <= ps_int_properties[I, index_nCount]
            action!(dimension, I, NI, ps_int_properties, ps_float_properties, parameters)
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
    device_interaction!(Backend, n_threads)(
        Dimension,
        particle_system.device_base_.is_alive_,
        particle_system.device_base_.int_properties_,
        particle_system.device_base_.float_properties_,
        particle_system.parameters_,
        particle_system.basic_index_.nCount,
        action!,
        ndrange = (Class.get_n_particles(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end
