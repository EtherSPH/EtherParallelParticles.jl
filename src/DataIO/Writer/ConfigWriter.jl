#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/15 21:28:38
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@kwdef struct ConfigWriter <: AbstractWriter
    path_::String = "config"
end

@inline function get_int_file_name(writer::ConfigWriter)::String
    return joinpath(writer.path_, "int.csv")
end

@inline function get_int_head_file_name(writer::ConfigWriter)::String
    return joinpath(writer.path_, "int_head.csv")
end

@inline function get_float_file_name(writer::ConfigWriter)::String
    return joinpath(writer.path_, "float.csv")
end

@inline function get_float_head_file_name(writer::ConfigWriter)::String
    return joinpath(writer.path_, "float_head.csv")
end

@inline function get_parameter_file_name(writer::ConfigWriter)::String
    return joinpath(writer.path_, "parameter.csv")
end

@inline function save(
    writer::ConfigWriter,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    int_file_name = get_int_file_name(writer)
    int_dataframe = DataFrames.DataFrame(
        name = particle_system.named_index_.int_named_index_table_.name_symbol_list_,
        capacity = collect(values(particle_system.named_index_.int_named_index_table_.capacity_named_tuple_)),
        index = collect(values(particle_system.named_index_.int_named_index_table_.index_named_tuple_)),
    )
    CSV.write(int_file_name, int_dataframe; delim = '\t')
    int_head_file_name = get_int_head_file_name(writer)
    int_head_dataframe = DataFrames.DataFrame(
        name = particle_system.named_index_.int_named_index_table_.name_string_table_head_,
        index = collect(1:length(particle_system.named_index_.int_named_index_table_.name_string_table_head_)),
    )
    CSV.write(int_head_file_name, int_head_dataframe; delim = '\t')
    float_file_name = get_float_file_name(writer)
    float_dataframe = DataFrames.DataFrame(
        name = particle_system.named_index_.float_named_index_table_.name_symbol_list_,
        capacity = collect(values(particle_system.named_index_.float_named_index_table_.capacity_named_tuple_)),
        index = collect(values(particle_system.named_index_.float_named_index_table_.index_named_tuple_)),
    )
    CSV.write(float_file_name, float_dataframe; delim = '\t')
    float_head_file_name = get_float_head_file_name(writer)
    float_head_dataframe = DataFrames.DataFrame(
        name = particle_system.named_index_.float_named_index_table_.name_string_table_head_,
        index = collect(1:length(particle_system.named_index_.float_named_index_table_.name_string_table_head_)),
    )
    CSV.write(float_head_file_name, float_head_dataframe; delim = '\t')
    parameter_file_name = get_parameter_file_name(writer)
    parameter_dict = Utility.convertPureNumberFromNamedTupleToDict(particle_system.basic_parameters_)
    parameter_dataframe =
        DataFrames.DataFrame(name = collect(keys(parameter_dict)), value = collect(values(parameter_dict)))
    CSV.write(parameter_file_name, parameter_dataframe; delim = '\t')
    return nothing
end
