#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/14 15:37:18
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

const kBackendDict = Dict(
    "cpu" => ("Array", "KernelAbstractions.CPU()"),
    "cuda" => ("CUDA.CuArray", "CUDA.CUDABackend()"),
    "rocm" => ("AMDGPU.ROCArray", "AMDGPU.ROCBackend()"),
    "oneapi" => ("oneAPI.oneArray", "oneAPI.oneAPIBackend()"),
    "metal" => ("Metal.MtlArray", "Metal.MetalBackend()"),
)

@inline function parallel(config_dict::AbstractDict)::Expr
    IT = config_dict["parallel"]["int"]
    FT = config_dict["parallel"]["float"]
    CT = kBackendDict[config_dict["parallel"]["backend"]][1]
    Backend = kBackendDict[config_dict["parallel"]["backend"]][2]
    Device = config_dict["parallel"]["device"]
    return Meta.parse(
        "const parallel = EtherParallelParticles.Environment.Parallel{$IT, $FT, $CT, $Backend}(); KernelAbstractions.device!($Backend, $Device)",
    )
end

@inline function domain(
    config_dict::AbstractDict,
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    if config_dict["domain"]["dimension"] == 2
        return Class.Domain2D{IT, FT}(
            config_dict["domain"]["gap"],
            config_dict["domain"]["first_x"],
            config_dict["domain"]["first_y"],
            config_dict["domain"]["last_x"],
            config_dict["domain"]["last_y"],
        )
    end
    # TODO: add 3D support
end

@inline function ParticleSystem(
    config_dict::AbstractDict,
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    int_named_tuple = Utility.convertDictToNamedTuple(config_dict["particle_system"]["int_named_tuple"])
    float_named_tuple = Utility.convertDictToNamedTuple(config_dict["particle_system"]["float_named_tuple"])
    basic_parameters = Utility.convertDictToNamedTuple(config_dict["particle_system"]["basic_parameters"])
    basic_index_map_dict = Utility.convertStringToSymbol(config_dict["particle_system"]["basic_index_map_dict"])
    if config_dict["particle_system"]["load"]["mode"] == "initial"
        n_particles = config_dict["particle_system"]["n_particles"]
        f = eval(Meta.parse(config_dict["particle_system"]["capacity_expand"]))
        n_capacity = Base.invokelatest(f, n_particles)
    else
        file = jldopen(config_dict["particle_system"]["load"]["file"], "r")
        is_movable = file["data/is_movable"]
        int_properties = file["data/int_properties"]
        float_properties = file["data/float_properties"]
        close(file)
        n_particles = length(is_movable)
        f = eval(Meta.parse(config_dict["particle_system"]["capacity_expand"]))
        n_capacity = Base.invokelatest(f, n_particles)
    end
    particle_system = Class.ParticleSystem(
        parallel,
        domain,
        n_particles,
        n_capacity,
        int_named_tuple,
        float_named_tuple,
        basic_parameters;
        basic_index_map_dict = basic_index_map_dict,
    )
    if config_dict["particle_system"]["load"]["mode"] == "continue"
        Class.set_is_movable!(particle_system, is_movable)
        Class.set_int_properties!(particle_system, int_properties)
        Class.set_float_properties!(particle_system, float_properties)
        Class.set_n_particles!(particle_system, n_particles)
    end
    return particle_system
end

@inline function NeighbourSystem(
    config_dict::AbstractDict,
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    active_pair = [v[1] => v[2] for v in config_dict["neighbour_system"]["active_pair"]]
    if config_dict["neighbour_system"]["periodic_boundary_policy"]["type"] == "none"
        periodic_boundary_policy = Class.NonePeriodicBoundaryPolicy
    else
        axis = config_dict["neighbour_system"]["periodic_boundary_policy"]["axis"]
        if N == 2
            periodic_boundary_policy = Class.PeriodicBoundaryPolicy2D{axis[1], axis[2]}
        else
            periodic_boundary_policy = Class.PeriodicBoundaryPolicy3D{axis[1], axis[2], axis[3]}
        end
    end
    max_neighbour_number = config_dict["neighbour_system"]["max_neighbour_number"]
    n_threads = config_dict["neighbour_system"]["n_threads"]
    return Class.NeighbourSystem(
        parallel,
        domain,
        active_pair,
        periodic_boundary_policy;
        max_neighbour_number = max_neighbour_number,
        n_threads = n_threads,
    )
end

@inline function Writer(config_dict::AbstractDict)::Writer
    return Writer(
        config_dict["writer"]["path"];
        file_name = config_dict["writer"]["file_name"],
        connect = config_dict["writer"]["connect"],
        digits = config_dict["writer"]["digits"],
    )
end

@inline template() = JSON.parsefile(joinpath(@__DIR__, "template.json"); dicttype = OrderedDict)

@inline function generateConfigDict(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, PeriodicBoundaryPolicy},
    writer::Writer;
    config_dict = template(),
) where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundaryPolicy,
}
    return config_dict
end
