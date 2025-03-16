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
    println(io, "    raw: $(get_path(writer.raw_writer_))")
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

@inline function wait(writer::Writer)::Nothing
    progress = ProgressMeter.Progress(
        length(writer.tasks_);
        dt = 0.5,
        desc = "Wait for writing finished...",
        barglyphs = BarGlyphs("[=> ]"),
        barlen = 50,
        color = :blue,
    )
    @info "Writing raw data with $(length(writer.tasks_)) tasks..."
    for i in eachindex(writer.tasks_)
        Base.fetch(writer.tasks_[i])
        ProgressMeter.update!(progress, i)
    end
    @info "Writing raw data finished."
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
    task = Threads.@spawn begin
        save(writer.raw_writer_, particle_system, time; appendix = appendix, format = format, compress = compress)
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
