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

@inline function assuredir(path::String)::Nothing
    if !isdir(path)
        mkpath(path)
    end
    return nothing
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
