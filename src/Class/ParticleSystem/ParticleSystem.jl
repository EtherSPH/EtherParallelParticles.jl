#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/11 15:07:23
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

include("ParticleSystemBase.jl")
include("NamedIndex.jl")
include("BasicIndex.jl")

@inline function defaultCapacityExpand(n_particles::IT)::IT where {IT <: Integer}
    return n_particles
end

abstract type AbstractParticleSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
} end

struct ParticleSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
} <: AbstractParticleSystem{IT, FT, CT, Backend, Dimension}
    host_base_::ParticleSystemBase{IT, FT}
    device_base_::ParticleSystemBase{IT, FT}
    named_index_::NamedIndex{IT}
    basic_index_::NamedTuple
    basic_parameters_::NamedTuple
    parameters_::NamedTuple # all combined things, including `named_index_.combined_index_named_tuple_` and `basic_parameters_`
end

@inline function ParticleSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    ::AbstractDomain{IT, FT, Dimension},
    n_particles::Integer,
    n_capacity::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple,
    basic_parameters::NamedTuple;
    basic_index_map_dict::AbstractDict{Symbol, Symbol} = kBasicIndexMapDict,
)::ParticleSystem{
    IT,
    FT,
    CT,
    Backend,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_particles = parallel(n_particles)
    n_capacity = parallel(n_capacity)
    int_named_tuple = parallel(int_named_tuple)
    float_named_tuple = parallel(float_named_tuple)
    basic_parameters = parallel(basic_parameters)
    cpu_parallel = Environment.Parallel{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend}()
    named_index = NamedIndex{IT}(int_named_tuple, float_named_tuple)
    n_int_capacity = get_n_int_capacity(named_index)
    n_float_capacity = get_n_float_capacity(named_index)
    host_base = ParticleSystemBase(cpu_parallel, n_capacity, n_int_capacity, n_float_capacity)
    device_base = ParticleSystemBase(parallel, n_capacity, n_int_capacity, n_float_capacity)
    basic_index =
        mapBasicIndex(parallel, get_index_named_tuple(named_index); basic_index_map_dict = basic_index_map_dict)
    parameters = merge(basic_parameters, named_index.combined_index_named_tuple_)
    host_base.n_particles_[1] = n_particles
    host_base.is_alive_[1:n_particles] .= IT(1)
    particle_system = ParticleSystem{IT, FT, CT, Backend, Dimension}(
        host_base,
        device_base,
        named_index,
        basic_index,
        basic_parameters,
        parameters,
    )
    toDevice!(particle_system)
    return particle_system
end

@inline function ParticleSystem(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    n_particles::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple,
    basic_parameters::NamedTuple;
    capacityExpand::Function = defaultCapacityExpand,
    basic_index_map_dict::AbstractDict{Symbol, Symbol} = kBasicIndexMapDict,
)::ParticleSystem{
    IT,
    FT,
    CT,
    Backend,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return ParticleSystem(
        parallel,
        domain,
        n_particles,
        capacityExpand(n_particles),
        int_named_tuple,
        float_named_tuple,
        basic_parameters,
        basic_index_map_dict = basic_index_map_dict,
    )
end

@inline function get_n_particles(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds return particle_system.host_base_.n_particles_[1]
end

@inline function get_alive_n_particles(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return sum(particle_system.host_base_.is_alive_)
end

@inline function get_n_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return length(particle_system.host_base_.is_alive_)
end

@inline function get_n_int_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return size(particle_system.host_base_.int_properties_, 2)
end

@inline function get_n_float_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return size(particle_system.host_base_.float_properties_, 2)
end

@inline function get_int_property(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return particle_system.named_index_.int_named_index_table_.capacity_named_tuple_
end

@inline function get_float_property(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return particle_system.named_index_.float_named_index_table_.capacity_named_tuple_
end

@inline function get_basic_index(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return particle_system.basic_index_
end

@inline function get_basic_parameters(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return particle_system.basic_parameters_
end

function Base.show(
    io::IO,
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension}
    println(io, "ParticleSystem{$IT, $FT, $CT, $Backend, $Dimension}(")
    println(io, "    n_particles: $(get_n_particles(particle_system))")
    println(io, "    n_capacity: $(get_n_capacity(particle_system))")
    println(io, "    n_alive particles: $(get_alive_n_particles(particle_system))")
    println(io, "    n_int_capacity: $(get_n_int_capacity(particle_system))")
    println(io, "    n_float_capacity: $(get_n_float_capacity(particle_system))")
    println(io, "    int_property: $(get_int_property(particle_system))")
    println(io, "    float_property: $(get_float_property(particle_system))")
    println(io, "    basic_index: $(get_basic_index(particle_system))")
    println(io, "    basic_parameters: $(get_basic_parameters(particle_system))")
    println(io, ")")
end

@inline function toDevice!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    transfer!(Backend, particle_system.device_base_, particle_system.host_base_)
    return nothing
end

@inline function toHost!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    transfer!(Backend, particle_system.host_base_, particle_system.device_base_)
    return nothing
end

@inline function set_n_particles!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
    n_particles::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds particle_system.host_base_.n_particles_[1] = IT(n_particles)
    fill!(particle_system.host_base_.is_alive_, IT(0))
    particle_system.host_base_.is_alive_[1:n_particles] .= IT(1)
    return nothing
end

@inline function set_is_movable!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
    is_movable::Array{<:Integer, 1},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_particles = get_n_particles(particle_system)
    @assert n_particles <= get_n_capacity(particle_system)
    @inbounds particle_system.host_base_.is_movable_[1:n_particles] .= IT.(is_movable)
    return nothing
end

@inline function set_int_properties!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
    int_properties::Array{<:Integer, 2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_particles = size(int_properties, 1)
    n_int_capacity = size(int_properties, 2)
    @assert n_particles <= get_n_capacity(particle_system)
    @assert n_int_capacity == get_n_int_capacity(particle_system)
    @inbounds particle_system.host_base_.int_properties_[1:n_particles, :] .= IT.(int_properties)
    return nothing
end

@inline function set_float_properties!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
    float_properties::Array{<:AbstractFloat, 2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_particles = size(float_properties, 1)
    n_float_capacity = size(float_properties, 2)
    @assert n_particles <= get_n_capacity(particle_system)
    @assert n_float_capacity == get_n_float_capacity(particle_system)
    @inbounds particle_system.host_base_.float_properties_[1:n_particles, :] .= FT.(float_properties)
    return nothing
end
