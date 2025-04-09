#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/20 18:01:55
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

const AbstractHostParticleSystem{IT, FT, Dimension} =
    AbstractParticleSystem{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend, Dimension}

# * ===================== HostParticleSystem Data Manipulation ===================== * #

@inline function HostParticleSystem{IT, FT, Dimension}(
    n_capacity::Integer,
    named_index::NamedIndex{IT},
    basic_parameters::NamedTuple;
    basic_index_map_dict::AbstractDict{Symbol, Symbol} = kBasicIndexMapDict,
)::HostParticleSystem{IT, FT, Dimension} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    parallel = Environment.Parallel{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend}()
    n_particles = parallel(0)
    n_capacity = parallel(n_capacity)
    basic_parameters = parallel(basic_parameters)
    n_int_capacity = get_n_int_capacity(named_index)
    n_float_capacity = get_n_float_capacity(named_index)
    base = ParticleSystemBase(parallel, n_capacity, n_int_capacity, n_float_capacity)
    basic_index =
        mapBasicIndex(parallel, get_index_named_tuple(named_index); basic_index_map_dict = basic_index_map_dict)
    parameters = merge(basic_parameters, named_index.combined_index_named_tuple_)
    Base.copyto!(base.n_particles_, [n_particles])
    particle_system = HostParticleSystem{IT, FT, Dimension}(
        [n_particles],
        base,
        named_index,
        basic_index,
        basic_parameters,
        parameters,
    )
    return particle_system
end

@inline function HostParticleSystem{IT, FT, Dimension}(
    n_capacity::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple,
    basic_parameters::NamedTuple;
    basic_index_map_dict::AbstractDict{Symbol, Symbol} = kBasicIndexMapDict,
)::HostParticleSystem{IT, FT, Dimension} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    parallel = Environment.Parallel{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend}()
    int_named_tuple = parallel(int_named_tuple)
    float_named_tuple = parallel(float_named_tuple)
    named_index = NamedIndex{IT}(int_named_tuple, float_named_tuple)
    return HostParticleSystem{IT, FT, Dimension}(
        n_capacity,
        named_index,
        basic_parameters,
        basic_index_map_dict = basic_index_map_dict,
    )
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    int_properties::Array{<:Integer, 2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = size(int_properties, 1)
    n_capacity = get_n_capacity(particle_system)
    @assert n_particles <= n_capacity "particles number exceed capacity, consider expand the capacity $(n_particles) > $(n_capacity)"
    @assert size(int_properties, 2) == get_n_int_capacity(particle_system) "int capacity not match $(size(int_properties, 2)) != $(get_n_int_capacity(particle_system))"
    @inbounds particle_system.base_.int_properties_[1:n_particles, :] .= IT.(int_properties)
    set_n_particles!(particle_system, n_particles)
    return nothing
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    float_properties::Array{<:AbstractFloat, 2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = size(float_properties, 1)
    n_capacity = get_n_capacity(particle_system)
    @assert n_particles <= n_capacity "particles number exceed capacity, consider expand the capacity $(n_particles) > $(n_capacity)"
    @assert size(float_properties, 2) == get_n_float_capacity(particle_system) "float capacity not match $(size(float_properties, 2)) != $(get_n_float_capacity(particle_system))"
    @inbounds particle_system.base_.float_properties_[1:n_particles, :] .= FT.(float_properties)
    set_n_particles!(particle_system, n_particles)
    return nothing
end

@inline function Base.reshape(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    n_capacity::Integer,
)::typeof(particle_system) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert n_capacity > get_n_capacity(particle_system) "$n_capacity <= $(get_n_capacity(particle_system)), no need to reshape"
    new_particle_system = HostParticleSystem{IT, FT, Dimension}(
        n_capacity,
        deepcopy(particle_system.named_index_),
        deepcopy(particle_system.basic_parameters_),
    )
    @inbounds new_particle_system.n_particles_[1] = particle_system.n_particles_[1]
    @inbounds new_particle_system.base_.is_alive_[1:get_n_capacity(particle_system)] .= particle_system.base_.is_alive_
    @inbounds new_particle_system.base_.cell_index_[1:get_n_capacity(particle_system)] .=
        particle_system.base_.cell_index_
    @inbounds new_particle_system.base_.int_properties_[1:get_n_capacity(particle_system), :] .=
        particle_system.base_.int_properties_
    @inbounds new_particle_system.base_.float_properties_[1:get_n_capacity(particle_system), :] .=
        particle_system.base_.float_properties_
    set_n_particles!(new_particle_system)
    return new_particle_system
end

@inline function Base.merge(
    particle_system_1::AbstractHostParticleSystem{IT, FT, Dimension},
    particle_system_2::AbstractHostParticleSystem{IT, FT, Dimension},
)::typeof(particle_system_1) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert get_n_int_capacity(particle_system_1) == get_n_int_capacity(particle_system_2) "int capacity not match"
    @assert get_n_float_capacity(particle_system_1) == get_n_float_capacity(particle_system_2) "float capacity not match"
    n_capacity = get_n_capacity(particle_system_1) + get_n_capacity(particle_system_2)
    new_particle_system = Base.reshape(particle_system_1, n_capacity)
    n_particles_1 = get_n_particles(particle_system_1)
    n_particles_2 = get_n_particles(particle_system_2)
    start_index = n_particles_1 + 1
    end_index = n_particles_1 + n_particles_2
    @inbounds new_particle_system.base_.is_alive_[start_index:end_index] .=
        particle_system_2.base_.is_alive_[1:n_particles_2]
    @inbounds new_particle_system.base_.cell_index_[start_index:end_index] .=
        particle_system_2.base_.cell_index_[1:n_particles_2]
    @inbounds new_particle_system.base_.int_properties_[start_index:end_index, :] .=
        particle_system_2.base_.int_properties_[1:n_particles_2, :]
    @inbounds new_particle_system.base_.float_properties_[start_index:end_index, :] .=
        particle_system_2.base_.float_properties_[1:n_particles_2, :]
    set_n_particles!(new_particle_system, end_index)
    return new_particle_system
end

@inline Base.merge(particle_system...) = reduce(merge, particle_system)

@inline function set_is_alive!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @inbounds particle_system.base_.is_alive_[1:get_n_particles(particle_system)] .= IT(1)
    return nothing
end

@inline function get_int_capacity(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(particle_system.named_index_.int_named_index_table_.capacity_named_tuple_, name)
    return getfield(particle_system.named_index_.int_named_index_table_.capacity_named_tuple_, name)
end

@inline function get_float_capacity(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(particle_system.named_index_.float_named_index_table_.capacity_named_tuple_, name)
    return getfield(particle_system.named_index_.float_named_index_table_.capacity_named_tuple_, name)
end

@inline function get_int_index(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(particle_system.named_index_.int_named_index_table_.index_named_tuple_, name)
    return getfield(particle_system.named_index_.int_named_index_table_.index_named_tuple_, name)
end

@inline function get_float_index(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(particle_system.named_index_.float_named_index_table_.index_named_tuple_, name)
    return getfield(particle_system.named_index_.float_named_index_table_.index_named_tuple_, name)
end

@inline function get_int(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    n_particles = get_n_particles(particle_system)
    @assert n_particles > 0 "no particles in the system"
    if capacity == 1
        return particle_system.base_.int_properties_[1:n_particles, index]
    else
        return particle_system.base_.int_properties_[1:n_particles, index:(index + capacity - 1)]
    end
end

@inline function get_int(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
    i::Integer,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    return particle_system.base_.int_properties_[i, index:(index + capacity - 1)]
end

@inline function get_float(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    n_particles = get_n_particles(particle_system)
    @assert n_particles > 0 "no particles in the system"
    if capacity == 1
        return particle_system.base_.float_properties_[1:n_particles, index]
    else
        return particle_system.base_.float_properties_[1:n_particles, index:(index + capacity - 1)]
    end
end

@inline function get_float(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
    i::Integer,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    return particle_system.base_.float_properties_[i, index:(index + capacity - 1)]
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    int_property::Array{<:Integer},
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    n_particles = size(int_property, 1)
    @assert n_particles <= get_n_capacity(particle_system) "particles number exceed capacity, consider expand the capacity $(n_particles) > $(get_n_capacity(particle_system))"
    @inbounds particle_system.base_.int_properties_[1:n_particles, index:(index + capacity - 1)] .= IT.(int_property)
    return nothing
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    int_property::Integer,
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    n_particles = get_n_particles(particle_system)
    @inbounds particle_system.base_.int_properties_[1:n_particles, index:(index + capacity - 1)] .= IT(int_property)
    return nothing
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    int_property::Array{<:Integer},
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    n_capacity = get_n_capacity(particle_system)
    @assert n_capacity >= i "particles number exceed capacity, consider expand the capacity $(n_capacity) > $(get_n_capacity(particle_system))"
    @assert capacity >= length(int_property) "int capacity not match"
    @inbounds particle_system.base_.int_properties_[i, index:(index + length(int_property) - 1)] .= IT.(int_property)
    return nothing
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    int_property::Integer,
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    n_capacity = get_n_capacity(particle_system)
    @assert n_capacity >= i "particles number exceed capacity, consider expand the capacity $(n_capacity) > $(get_n_capacity(particle_system))"
    @inbounds particle_system.base_.int_properties_[i, index:(index + capacity - 1)] .= IT(int_property)
    return nothing
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    float_property::Array{<:AbstractFloat},
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    n_particles = size(float_property, 1)
    @assert n_particles <= get_n_capacity(particle_system) "particles number exceed capacity, consider expand the capacity $(n_particles) > $(get_n_capacity(particle_system))"
    @inbounds particle_system.base_.float_properties_[1:n_particles, index:(index + capacity - 1)] .=
        FT.(float_property)
    return nothing
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    float_property::AbstractFloat,
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    n_particles = get_n_particles(particle_system)
    @inbounds particle_system.base_.float_properties_[1:n_particles, index:(index + capacity - 1)] .= FT(float_property)
    return nothing
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    float_property::Array{<:AbstractFloat},
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    n_capacity = get_n_capacity(particle_system)
    @assert n_capacity >= i "particles number exceed capacity, consider expand the capacity $(n_capacity) > $(get_n_capacity(particle_system))"
    @assert capacity >= length(float_property) "float capacity not match"
    @inbounds particle_system.base_.float_properties_[i, index:(index + length(float_property) - 1)] .=
        FT.(float_property)
    return nothing
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    float_property::AbstractFloat,
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    n_capacity = get_n_capacity(particle_system)
    @assert n_capacity >= i "particles number exceed capacity, consider expand the capacity $(n_capacity) > $(get_n_capacity(particle_system))"
    @inbounds particle_system.base_.float_properties_[i, index:(index + capacity - 1)] .= FT(float_property)
    return nothing
end

@inline function Base.getindex(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    if name in keys(particle_system.named_index_.int_named_index_table_.index_named_tuple_)
        return get_int(particle_system, name)
    elseif name in keys(particle_system.named_index_.float_named_index_table_.index_named_tuple_)
        return get_float(particle_system, name)
    else
        error("unknown property $(name)")
    end
end

@inline function Base.setindex!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    value,
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    if name in keys(particle_system.named_index_.int_named_index_table_.index_named_tuple_)
        return set_int!(particle_system, value, name)
    elseif name in keys(particle_system.named_index_.float_named_index_table_.index_named_tuple_)
        return set_float!(particle_system, value, name)
    else
        error("unknown property $(name)")
    end
end

@inline function Base.setindex!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    value,
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    if name in keys(particle_system.named_index_.int_named_index_table_.index_named_tuple_)
        return set_int!(particle_system, value, name, i)
    elseif name in keys(particle_system.named_index_.float_named_index_table_.index_named_tuple_)
        return set_float!(particle_system, value, name, i)
    else
        error("unknown property $(name)")
    end
end

# * ===================== Particle Definition ===================== * #

abstract type AbstractParticle{IT <: Integer, FT <: AbstractFloat} end

struct Particle{IT <: Integer, FT <: AbstractFloat} <: AbstractParticle{IT, FT}
    int_::Vector{IT}
    float_::Vector{FT}
end

@inline function get_int_capacity(particle::AbstractParticle{IT, FT})::IT where {IT <: Integer, FT <: AbstractFloat}
    return length(particle.int_)
end

@inline function get_float_capacity(particle::AbstractParticle{IT, FT})::IT where {IT <: Integer, FT <: AbstractFloat}
    return length(particle.float_)
end

@inline function Particle(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
)::Particle{IT, FT} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_int_capacity = get_n_int_capacity(particle_system)
    n_float_capacity = get_n_float_capacity(particle_system)
    int_ = zeros(IT, n_int_capacity)
    float_ = zeros(FT, n_float_capacity)
    return Particle{IT, FT}(int_, float_)
end

@inline function Base.push!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = Atomix.@atomic particle_system.n_particles_[1] += IT(1) # ensure thread-safe
    n_capacity = get_n_capacity(particle_system)
    @assert n_particles <= n_capacity "particles number exceed capacity, consider expand the capacity $(n_particles) > $(n_capacity)"
    @assert get_n_int_capacity(particle_system) == get_int_capacity(particle) "int capacity not match $(get_n_int_capacity(particle_system)) != $(get_int_capacity(particle))"
    @assert get_n_float_capacity(particle_system) == get_float_capacity(particle) "float capacity not match $(get_n_float_capacity(particle_system)) != $(get_float_capacity(particle))"
    @inbounds particle_system.base_.is_alive_[n_particles] = IT(1)
    @inbounds particle_system.base_.int_properties_[n_particles, :] .= particle.int_
    @inbounds particle_system.base_.float_properties_[n_particles, :] .= particle.float_
    return nothing
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    value::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    @assert capacity == 1 "capacity not match"
    @inbounds particle.int_[index] = IT(value)
    return nothing
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    i::Integer,
    value::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    @assert capacity >= i "capacity not match"
    @inbounds particle.int_[index + i - 1] = IT(value)
    return nothing
end

@inline function get_int(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    i::Integer,
)::IT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    @assert capacity >= i "capacity not match"
    return particle.int_[index + i - 1]
end

@inline function set_int!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    value::Array{<:Integer},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    @assert capacity >= length(value) "capacity not match"
    @inbounds for i in 0:(length(value) - 1)
        particle.int_[index + i] = IT(value[i + 1])
    end
    return nothing
end

@inline function get_int(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_int_index(particle_system, name)
    capacity = get_int_capacity(particle_system, name)
    if capacity == 1
        return particle.int_[index]
    else
        return [particle.int_[index + i] for i in 0:(capacity - 1)]
    end
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    value::AbstractFloat,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    @assert capacity == 1 "capacity not match"
    @inbounds particle.float_[index] = FT(value)
    return nothing
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    i::Integer,
    value::AbstractFloat,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    @assert capacity >= i "capacity not match"
    @inbounds particle.float_[index + i - 1] = FT(value)
    return nothing
end

@inline function get_float(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    i::Integer,
)::FT where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    @assert capacity >= i "capacity not match"
    return particle.float_[index + i - 1]
end

@inline function set_float!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
    value::Array{<:AbstractFloat},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    @assert capacity >= length(value) "capacity not match"
    @inbounds for i in 0:(length(value) - 1)
        particle.float_[index + i] = FT(value[i + 1])
    end
    return nothing
end

@inline function get_float(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    particle::AbstractParticle{IT, FT},
    name::Symbol,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    index = get_float_index(particle_system, name)
    capacity = get_float_capacity(particle_system, name)
    if capacity == 1
        return particle.float_[index]
    else
        return [particle.float_[index + i] for i in 0:(capacity - 1)]
    end
end
