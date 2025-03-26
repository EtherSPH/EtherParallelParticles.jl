#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/15 21:33:04
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractRawWriter <: AbstractWriter end
@kwdef struct RawWriter <: AbstractRawWriter
    path_::String = "raw"
    file_name_::String = "result"
    connect_::String = "_"
    digits_::Int64 = 4
    suffix_::String = ".jld2"
end

@inline function appendix(; dicttype = OrderedDict)
    return dicttype("TMSTEP" => 0, "TimeValue" => 0.0, "TimeStamp" => "", "WriteStep" => 0)
end

@inline function get_file_name(writer::AbstractRawWriter, step::Integer)::String
    return joinpath(
        writer.path_,
        string(writer.file_name_, writer.connect_, string(step, pad = writer.digits_), writer.suffix_),
    )
end

function Base.show(io::IO, writer::RawWriter)
    println(io, "RawWriter:")
    println(io, "  format like: $(get_file_name(writer, 1))")
end

@inline function saveParticleSystem(
    jld_file::JLD2.JLDFile,
    particle_system::AbstractParticleSystem{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @inbounds n_particles = Class.get_n_particles(particle_system)
    @inbounds mask = particle_system.base_.is_alive_[1:n_particles] .== 1
    @inbounds write(jld_file, "raw/int_properties", particle_system.base_.int_properties_[1:n_particles, :][mask, :])
    @inbounds write(
        jld_file,
        "raw/float_properties",
        particle_system.base_.float_properties_[1:n_particles, :][mask, :],
    )
    return nothing
end

@inline function saveAppendix(jld_file::JLD2.JLDFile, dict::AbstractDict; format = "yyyy_mm_dd_HH_MM_SS")::Nothing
    dict["TimeStamp"] = Utility.timeStamp(; format = format)
    for (key, value) in dict
        write(jld_file, "appendix/$key", value)
    end
    return nothing
end
