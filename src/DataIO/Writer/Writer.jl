#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/15 15:51:01
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractWriter end

# https://juliaio.github.io/JLD2.jl/dev/compression/
# - CodecZlib.ZlibCompressor: the default as it is widely used
# - CodecBzip2.Bzip2Compressor: can often times be faster
# - CodecLz4.Lz4Compressor: fast, but not compatible to the LZ4 shipped by HDF5
# - CodecZstd.ZstdCompressor: fast, wide range of compression size vs speed trade-offs
# * memo: when using python, `h5py` with `hdf5plugin` are required at the same time to read the compressed data.
# ! warning: compress may largely slow down the output speed, in case that frequent output step is not recommended
const kDefaultCompressor = CodecZstd.ZstdFrameCompressor()

@inline function get_path(writer::AbstractWriter)::String
    return writer.path_
end

include("ConfigWriter.jl")
include("RawWriter.jl")

struct Writer <: AbstractWriter
    path_::String
    config_writer_::ConfigWriter
    raw_writer_::RawWriter
    tasks_::Vector{Base.Task}
end

function Base.show(io::IO, writer::Writer)
    println(io, "Writer:")
    println(io, "    root path: $(get_path(writer))")
    println(io, "    config: $(get_path(writer.config_writer_))")
    println(io, "    raw: $(get_path(writer.raw_writer_))  eg. $(get_file_name(writer.raw_writer_, 1))")
    println(io, "    write tasks: $(length(writer.tasks_))")
end

@inline function Writer(
    path::String;
    file_name::String = "result",
    connect::String = "_",
    digits::Int64 = 4,
    suffix::String = ".jld2",
)::Writer
    config_writer = ConfigWriter(joinpath(path, "config"))
    raw_writer = RawWriter(joinpath(path, "raw"), file_name, connect, digits, suffix)
    tasks = Task[]
    return Writer(path, config_writer, raw_writer, tasks)
end

@inline function Writer(path::String, config_writer::ConfigWriter, raw_writer::RawWriter)::Writer
    tasks = Task[]
    return Writer(path, config_writer, raw_writer, tasks)
end

@inline function get_raw_path(writer::AbstractWriter)::String
    return get_path(writer.raw_writer_)
end

@inline function get_raw_file(writer::AbstractWriter; step = :first)::String
    raw_files = get_raw_files(writer)
    if step == :first
        step = 1
    elseif step == :last
        step = length(raw_files)
    else
        step += 1
    end
    @assert step <= length(raw_files) "step out of range, max step is $(length(raw_files))"
    return raw_files[step]
end

@inline function get_raw_files(writer::AbstractWriter)::Vector{String}
    raw_path = get_raw_path(writer)
    files_list = readdir(raw_path)
    sort!(files_list)
    return [joinpath(raw_path, file) for file in files_list]
end

function Base.length(writer::AbstractWriter)
    return length(get_raw_files(writer))
end

@inline function assuredir(path::String)::Nothing
    if !isdir(path)
        mkpath(path)
    end
    return nothing
end

@inline function get_config_dict(writer::AbstractWriter; format = :json, dicttype = OrderedDict)
    config_file = get_config_file(writer; format = format)
    if format == :json
        config_dict = JSON.parsefile(config_file; dicttype = dicttype)
    elseif format == :yaml
        config_dict = YAML.load_file(config_file; dicttype = dicttype)
    else
        @warn "unknown config file format, use json as default"
        config_dict = JSON.parsefile(config_file; dicttype = dicttype)
    end
    return config_dict
end

@inline function get_config_file(writer::AbstractWriter; format = :json)::String
    path = get_path(writer.config_writer_)
    if format == :json
        file_name = "config.json"
    elseif format == :yaml
        file_name = "config.yaml"
    else
        @warn "unknown config file format, use json as default"
        file_name = "config.json"
    end
    return joinpath(path, file_name)
end

@inline function mkdir(writer::AbstractWriter)::Nothing
    for path in [get_path(writer), get_path(writer.config_writer_), get_path(writer.raw_writer_)]
        assuredir(path)
    end
    return nothing
end

@inline function cleandir(writer::AbstractWriter)::Nothing
    if !isdir(get_path(writer))
        @info "directory $(get_path(writer)) not exist."
        return nothing
    else
        @warn "remove all files & folders recursively in $(get_path(writer))"
        rm(get_path(writer); force = true, recursive = true)
        return nothing
    end
    return nothing
end

@inline function rmdir(writer::AbstractWriter)::Nothing
    cleandir(writer)
    rm(get_path(writer); force = true)
    return nothing
end

@inline function wait!(writer::AbstractWriter)::Nothing
    if isempty(writer.tasks_)
        return nothing
    else
        @inbounds Base.fetch(writer.tasks_[end]) # wait for the last task finished
    end
    return nothing
end

@inline function save!(
    writer::AbstractWriter,
    particle_system::AbstractParticleSystem{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend, Dimension},
    appendix::AbstractDict;
    format = "yyyy_mm_dd_HH_MM_SS",
    compress = kDefaultCompressor,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    jld_file_name = get_file_name(writer.raw_writer_, appendix["WriteStep"])
    jld_file = JLD2.jldopen(jld_file_name, "w"; compress = compress)
    saveAppendix(jld_file, appendix; format = format)
    task = Threads.@spawn begin
        saveParticleSystem(jld_file, particle_system)
        close(jld_file)
    end
    push!(writer.tasks_, task)
    return nothing
end

@inline function syncsave!(
    writer::AbstractWriter,
    host_particle_system::AbstractParticleSystem{
        IT,
        FT,
        Environment.kCPUContainerType,
        Environment.kCPUBackend,
        Dimension,
    },
    device_particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    appendix::AbstractDict;
    format = "yyyy_mm_dd_HH_MM_SS",
    compress = kDefaultCompressor,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    wait!(writer)
    Class.syncto!(host_particle_system, device_particle_system)
    save!(writer, host_particle_system, appendix; format = format, compress = compress)
    return nothing
end

@inline function asyncsave!(
    writer::AbstractWriter,
    host_particle_system::AbstractParticleSystem{
        IT,
        FT,
        Environment.kCPUContainerType,
        Environment.kCPUBackend,
        Dimension,
    },
    device_particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    appendix::AbstractDict;
    format = "yyyy_mm_dd_HH_MM_SS",
    compress = kDefaultCompressor,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    wait!(writer)
    Class.syncto!(host_particle_system, device_particle_system)
    save!(writer, host_particle_system, appendix; format = format, compress = compress)
    return nothing
end

@inline function load!(
    writer::AbstractWriter,
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension};
    appendix::AbstractDict = appendix(),
    step = :first,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    file_name = get_raw_file(writer; step = step)
    JLD2.jldopen(file_name, "r") do jld_file
        for key in keys(appendix)
            appendix[key] = jld_file["appendix/$key"]
        end
        Class.set_int!(particle_system, jld_file["raw/int_properties"])
        Class.set_float!(particle_system, jld_file["raw/float_properties"])
    end
    Class.set_is_alive!(particle_system)
    return nothing
end

@inline function load(writer::AbstractWriter; appendix::AbstractDict = appendix(), step = :first)
    config_dict = JSON.parsefile(get_config_file(writer; format = :json); dicttype = OrderedDict)
    IT = eval(Meta.parse(config_dict["parallel"]["int"]))
    FT = eval(Meta.parse(config_dict["parallel"]["float"]))
    parallel = Environment.Parallel{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend}()
    domain = DataIO.Domain(config_dict, parallel)
    host_particle_system = DataIO.ParticleSystem(config_dict, parallel, domain)
    load!(writer, host_particle_system; appendix = appendix, step = step)
    return host_particle_system
end
