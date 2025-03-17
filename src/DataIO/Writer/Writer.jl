#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/15 15:51:01
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractWriter end

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

@inline function Writer(path::String; file_name::String = "result", connect::String = "_", digits::Int64 = 4)::Writer
    config_writer = ConfigWriter(joinpath(path, "config"))
    raw_writer = RawWriter(joinpath(path, "raw"), file_name, connect, digits)
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

@inline function mkdir(writer::Writer)::Nothing
    for path in [get_path(writer), get_path(writer.config_writer_), get_path(writer.raw_writer_)]
        assuredir(path)
    end
    return nothing
end

@inline function cleandir(writer::Writer)::Nothing
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

@inline function rmdir(writer::Writer)::Nothing
    cleandir(writer)
    rm(get_path(writer); force = true)
    return nothing
end

@inline function wait(writer::Writer)::Nothing
    @inbounds Base.fetch(writer.tasks_[end]) # wait for the last task finished
    return nothing
end

@inline function save(
    writer::Writer,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    time::AbstractDict;
    appendix::AbstractDict = Dict(),
    format = "yyyy_mm_dd_HH_MM_SS",
    compress = CodecZlib.ZlibCompressor(),
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    jld_file_name = get_file_name(writer.raw_writer_, time["WriteStep"])
    if isempty(writer.tasks_)
        nothing
    else
        wait(writer)
    end
    task = Threads.@spawn begin
        save(jld_file_name, particle_system, time; appendix = appendix, format = format, compress = compress)
    end
    push!(writer.tasks_, task)
    return nothing
end

@inline function save(
    writer::Writer,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    save(writer.config_writer_, particle_system)
    return nothing
end
