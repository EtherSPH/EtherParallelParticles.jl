#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/18 17:42:10
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

const kBasicIndexMapDict = Dict{Symbol, Symbol}(
    :PositionVec => :PositionVec,
    :Tag => :Tag,
    :nCount => :nCount,
    :nIndex => :nIndex,
    :nRVec => :nRVec,
    :nR => :nR,
)

@inline function mapBasicIndex(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    named_index_tuple::NamedTuple;
    basic_index_map_dict::AbstractDict{Symbol, Symbol} = kBasicIndexMapDict,
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    symbol_to_symbol_dict = Dict{Symbol, Symbol}()
    for (key, value) in kBasicIndexMapDict
        if haskey(basic_index_map_dict, key)
            symbol_to_symbol_dict[key] = basic_index_map_dict[key]
        else
            symbol_to_symbol_dict[key] = value
        end
    end
    name_symbol_list = Symbol.(keys(symbol_to_symbol_dict))
    index_symbol_list = zeros(IT, length(name_symbol_list))
    for i in eachindex(index_symbol_list)
        name_symbol = name_symbol_list[i]
        mapped_name_symbol = symbol_to_symbol_dict[name_symbol]
        @assert haskey(named_index_tuple, mapped_name_symbol)
        name_index = named_index_tuple[mapped_name_symbol]
        index_symbol_list[i] = name_index
    end
    basic_index = NamedTuple{Tuple(name_symbol_list)}(index_symbol_list)
    return parallel(basic_index)
end
