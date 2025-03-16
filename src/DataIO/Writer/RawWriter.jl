#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/15 21:33:04
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@kwdef struct RawWriter <: AbstractWriter
    path_::String = "raw"
    file_name_::String = "result"
    connect_::String = "_"
    digits_::Int64 = 4
end

@inline function timedict(; dicttype = OrderedDict)
    return dicttype("TMSTEP" => 0, "TimeValue" => 0.0, "TimeStamp" => "", "WriteStep" => 0)
end

@inline function get_file_name(writer::RawWriter, step::Int64)::String
    return joinpath(
        writer.path_,
        string(writer.file_name_, writer.connect_, string(step, pad = writer.digits_), ".jld2"),
    )
end

function Base.show(io::IO, writer::RawWriter)
    println(io, "RawWriter:")
    println(io, "  format like: $(get_file_name(writer, 1))")
end

@inline function saveParticleSystem(
    jld_file::JLD2.JLDFile,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds mask = particle_system.host_base_.is_alive_ .== 1
    @inbounds jld_file["raw/is_movable"] = particle_system.host_base_.is_movable_[mask]
    @inbounds jld_file["raw/int_properties"] = particle_system.host_base_.int_properties_[mask, :]
    @inbounds jld_file["raw/float_properties"] = particle_system.host_base_.float_properties_[mask, :]
    return nothing
end

@inline function saveTime(jld_file::JLD2.JLDFile, dict::AbstractDict; format = "yyyy_mm_dd_HH_MM_SS")::Nothing
    dict["TimeStamp"] = Utility.timeStamp(; format = format)
    for (key, value) in dict
        jld_file[string("time/", key)] = value
    end
    return nothing
end

@inline function saveAppendix(jld_file::JLD2.JLDFile, dict::AbstractDict)::Nothing
    for (key, value) in dict
        jld_file[string("appendix/", key)] = value
    end
    return nothing
end

@inline function save(
    writer::RawWriter,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    time::AbstractDict;
    appendix::AbstractDict = Dict(),
    format = "yyyy_mm_dd_HH_MM_SS",
    compress = CodecZlib.ZlibCompressor(),
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    jld_file = JLD2.jldopen(get_file_name(writer, time["WriteStep"]), "w"; compress = compress)
    saveParticleSystem(jld_file, particle_system)
    saveTime(jld_file, time; format = format)
    saveAppendix(jld_file, appendix)
    close(jld_file)
    return nothing
end
