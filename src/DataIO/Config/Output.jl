#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/16 17:59:40
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@inline template() = JSON.parsefile(joinpath(@__DIR__, "template.json"); dicttype = OrderedDict)

@inline function Base.replace!(
    config_dict::AbstractDict,
    ::AbstractParallel{IT, FT, CT, Backend},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    config_dict["parallel"]["int"] = "$IT"
    config_dict["parallel"]["float"] = "$FT"
    container_string = "$CT"
    name = Environment.kContainerToName[container_string]
    config_dict["parallel"]["backend"] = name
    device = KernelAbstractions.device(Backend)
    config_dict["parallel"]["device"] = device
    return nothing
end

@inline function Base.replace!(
    config_dict::AbstractDict,
    domain::AbstractDomain{IT, FT, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    config_dict["domain"]["dimension"] = N
    config_dict["domain"]["gap"] = Class.get_gap(domain)
    config_dict["domain"]["first_x"] = Class.get_first_x(domain)
    config_dict["domain"]["first_y"] = Class.get_first_y(domain)
    config_dict["domain"]["last_x"] = Class.get_last_x(domain)
    config_dict["domain"]["last_y"] = Class.get_last_y(domain)
    if N == 3
        config_dict["domain"]["first_z"] = Class.get_first_z(domain)
        config_dict["domain"]["last_z"] = Class.get_last_z(domain)
    end
    return nothing
end

@inline function Base.replace!(
    config_dict::AbstractDict,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    n_particles = Class.get_n_particles(particle_system)
    n_capacity = Class.get_n_capacity(particle_system)
    @assert n_particles <= n_capacity
    config_dict["particle_system"]["n_particles"] = n_particles
    config_dict["particle_system"]["capacity_expand"] = "n -> n + $(n_capacity - n_particles)"
    config_dict["particle_system"]["int_named_tuple"] = Utility.convertNamedTupleToDict(
        particle_system.named_index_.int_named_index_table_.capacity_named_tuple_;
        dicttype = OrderedDict,
    )
    config_dict["particle_system"]["float_named_tuple"] = Utility.convertNamedTupleToDict(
        particle_system.named_index_.float_named_index_table_.capacity_named_tuple_;
        dicttype = OrderedDict,
    )
    config_dict["particle_system"]["basic_parameters"] =
        Utility.convertPureNumberFromNamedTupleToDict(particle_system.basic_parameters_; dicttype = OrderedDict)
    return nothing
end

@inline function Base.replace!(
    config_dict::AbstractDict,
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    active_pair = [[ac.first, ac.second] for ac in neighbour_system.active_pair_.pair_vector_]
    config_dict["neighbour_system"]["active_pair"] = active_pair
    config_dict["neighbour_system"]["max_neighbour_number"] =
        size(neighbour_system.base_.contained_particle_index_list_, 2)
    if PeriodicBoundaryPolicy == Class.NonePeriodicBoundaryPolicy
        config_dict["neighbour_system"]["periodic_boundary_policy"]["type"] = "none"
    elseif PeriodicBoundaryPolicy <: Class.PeriodicBoundaryPolicy2D
        config_dict["neighbour_system"]["periodic_boundary_policy"]["type"] = "2D"
        config_dict["neighbour_system"]["periodic_boundary_policy"]["axis"][1] = PeriodicBoundaryPolicy.parameters[1]
        config_dict["neighbour_system"]["periodic_boundary_policy"]["axis"][2] = PeriodicBoundaryPolicy.parameters[2]
    elseif PeriodicBoundaryPolicy <: Class.PeriodicBoundaryPolicy3D
        config_dict["neighbour_system"]["periodic_boundary_policy"]["type"] = "3D"
        config_dict["neighbour_system"]["periodic_boundary_policy"]["axis"][1] = PeriodicBoundaryPolicy.parameters[1]
        config_dict["neighbour_system"]["periodic_boundary_policy"]["axis"][2] = PeriodicBoundaryPolicy.parameters[2]
        config_dict["neighbour_system"]["periodic_boundary_policy"]["axis"][3] = PeriodicBoundaryPolicy.parameters[3]
    end
    return nothing
end

@inline function Base.replace!(config_dict::AbstractDict, writer::Writer)::Nothing
    config_dict["writer"]["path"] = get_path(writer)
    config_dict["writer"]["file_name"] = writer.raw_writer_.file_name_
    config_dict["writer"]["connect"] = writer.raw_writer_.connect_
    config_dict["writer"]["digits"] = writer.raw_writer_.digits_
    config_dict["writer"]["suffix"] = writer.raw_writer_.suffix_
    return nothing
end

@inline function config(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    domain::AbstractDomain{IT, FT, Dimension},
    particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT3, Backend3, PeriodicBoundaryPolicy},
    writer::Writer;
    config_dict::AbstractDict = template(),
)::AbstractDict where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    Backend1,
    CT2 <: AbstractArray,
    Backend2,
    CT3 <: AbstractArray,
    Backend3,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    config_dict = deepcopy(config_dict)
    replace!(config_dict, parallel)
    replace!(config_dict, domain)
    replace!(config_dict, particle_system)
    replace!(config_dict, neighbour_system)
    replace!(config_dict, writer)
    return config_dict
end

@inline function writeConfig(
    writer::Writer,
    config_dict::AbstractDict,
    format::Symbol = :json, # :json or :yaml
)::Nothing
    if format == :json
        open(joinpath(get_path(writer.config_writer_), "config.json"), "w") do file
            JSON.print(file, config_dict, 4)
        end
        return nothing
    elseif format == :yaml
        YAML.write_file(joinpath(get_path(writer.config_writer_), "config.yaml"), config_dict)
        return nothing
    else
        @warn "Unsupported format: $format, defaulting to :json"
        writeConfig(writer, config_dict, :json)
        return nothing
    end
    return nothing
end

@inline function save(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    domain::AbstractDomain{IT, FT, Dimension},
    particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT3, Backend3, PeriodicBoundaryPolicy},
    writer::Writer;
    config_dict = template(),
    format::Symbol = :json, # :json, :yaml or :all
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    Backend1,
    CT2 <: AbstractArray,
    Backend2,
    CT3 <: AbstractArray,
    Backend3,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    saveConfig(writer.config_writer_, particle_system)
    config_dict = config(parallel, domain, particle_system, neighbour_system, writer; config_dict = config_dict)
    if format == :all
        for form in [:json, :yaml]
            writeConfig(writer, config_dict, form)
        end
    end
    return nothing
end
