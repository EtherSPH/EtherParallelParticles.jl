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

# * ===================== ParticleSystem Definition ===================== * #

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
    n_particles_::Vector{IT}
    base_::ParticleSystemBase{IT, FT, Backend}
    named_index_::NamedIndex{IT}
    basic_index_::NamedTuple
    basic_parameters_::NamedTuple
    parameters_::NamedTuple # all combined things, including `named_index_.combined_index_named_tuple_` and `basic_parameters_`
end

const HostParticleSystem{IT, FT, Dimension} =
    ParticleSystem{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend, Dimension}

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
    named_index = NamedIndex{IT}(int_named_tuple, float_named_tuple)
    n_int_capacity = get_n_int_capacity(named_index)
    n_float_capacity = get_n_float_capacity(named_index)
    base = ParticleSystemBase(parallel, n_capacity, n_int_capacity, n_float_capacity)
    basic_index =
        mapBasicIndex(parallel, get_index_named_tuple(named_index); basic_index_map_dict = basic_index_map_dict)
    parameters = merge(basic_parameters, named_index.combined_index_named_tuple_)
    Base.copyto!(base.n_particles_, [n_particles])
    is_alive = zeros(IT, n_capacity)
    is_alive[1:n_particles] .= IT(1)
    Base.copyto!(base.is_alive_, is_alive)
    particle_system = ParticleSystem{IT, FT, CT, Backend, Dimension}(
        [n_particles],
        base,
        named_index,
        basic_index,
        basic_parameters,
        parameters,
    )
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

@inline function count!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    Base.copyto!(particle_system.n_particles_, particle_system.base_.n_particles_)
    return nothing
end

@inline function set_n_particles!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    Base.copyto!(particle_system.base_.n_particles_, particle_system.n_particles_)
    return nothing
end

@inline function set_n_particles!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    n_particles::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds particle_system.n_particles_[1] = IT(n_particles)
    set_n_particles!(particle_system)
    return nothing
end

@inline function clean!(
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds particle_system.n_particles_[1] = IT(0)
    set_n_particles!(particle_system)
    return nothing
end

@inline function get_n_particles(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds return particle_system.n_particles_[1]
end

@inline function get_alive_n_particles(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return sum(particle_system.base_.is_alive_)
end

@inline function get_n_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return length(particle_system.base_.is_alive_)
end

@inline function get_n_int_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return size(particle_system.base_.int_properties_, 2)
end

@inline function get_n_float_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return size(particle_system.base_.float_properties_, 2)
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

# * ===================== ParticleSystem Data Transfer ===================== * #

@inline function mirror(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2, Dimension},
)::ParticleSystem{
    IT,
    FT,
    CT1,
    Backend1,
    Dimension,
} where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    CT2 <: AbstractArray,
    Backend1,
    Backend2,
    Dimension <: AbstractDimension,
}
    n_particles = deepcopy(particle_system.n_particles_)
    n_capacity = get_n_capacity(particle_system)
    base = ParticleSystemBase(
        parallel,
        n_capacity,
        get_n_int_capacity(particle_system),
        get_n_float_capacity(particle_system),
    )
    syncto!(base, particle_system.base_)
    named_index = deepcopy(particle_system.named_index_)
    basic_index = deepcopy(particle_system.basic_index_)
    basic_parameters = deepcopy(particle_system.basic_parameters_)
    parameters = deepcopy(particle_system.parameters_)
    return ParticleSystem{IT, FT, CT1, Backend1, Dimension}(
        n_particles,
        base,
        named_index,
        basic_index,
        basic_parameters,
        parameters,
    )
end

@inline function mirror(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::ParticleSystem{
    IT,
    FT,
    Environment.kCPUContainerType,
    Environment.kCPUBackend,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    parallel = Environment.Parallel{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend}()
    return mirror(parallel, particle_system)
end

@inline function syncto!(
    destination_particle_system::AbstractParticleSystem{IT, FT, CT1, Backend1, Dimension},
    source_particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2, Dimension},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    CT2 <: AbstractArray,
    Backend1,
    Backend2,
    Dimension <: AbstractDimension,
}
    @inbounds destination_particle_system.n_particles_[1] = source_particle_system.n_particles_[1]
    syncto!(destination_particle_system.base_, source_particle_system.base_)
    return nothing
end

@inline function asyncto!(
    destination_particle_system::AbstractParticleSystem{IT, FT, CT1, Backend1, Dimension},
    source_particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2, Dimension},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    CT2 <: AbstractArray,
    Backend1,
    Backend2,
    Dimension <: AbstractDimension,
}
    @inbounds destination_particle_system.n_particles_[1] = source_particle_system.n_particles_[1]
    asyncto!(destination_particle_system.base_, source_particle_system.base_)
    return nothing
end

@inline to!(dst, src) = syncto!(dst, src) # default `to!` as `syncto!`

include("HostParticleSystem.jl")
