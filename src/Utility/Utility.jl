#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/20 16:54:27
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Utility

using Dates
using OrderedCollections

@inline function convertSymbolToString(dict::AbstractDict; dicttype = OrderedDict)::AbstractDict
    key_s = String.(keys(dict))
    value_s = String.(values(dict))
    return dicttype(key_s .=> value_s)
end

@inline function convertStringToSymbol(dict::AbstractDict; dicttype = OrderedDict)::AbstractDict
    key_s = Symbol.(keys(dict))
    value_s = Symbol.(values(dict))
    return dicttype(key_s .=> value_s)
end

@inline function convertNamedTupleToDict(named_tuple::NamedTuple; dicttype = OrderedDict)::AbstractDict
    key_s = String.(keys(named_tuple))
    value_s = values(named_tuple)
    @assert dicttype in [Dict, OrderedDict]
    return dicttype(key_s .=> value_s)
end

@inline function convertDictToNamedTuple(dict::AbstractDict)::NamedTuple
    key_s = Tuple(Symbol.(keys(dict)))
    value_s = Tuple(values(dict))
    return NamedTuple{key_s}(value_s)
end

@inline function convertPureNumberFromNamedTupleToDict(named_tuple::NamedTuple; dicttype = OrderedDict)::AbstractDict
    key_s = collect(keys(named_tuple))
    value_s = collect(values(named_tuple))
    pure_keys = Symbol[]
    pure_values = []
    for i in eachindex(value_s)
        if typeof(value_s[i]) <: Real
            push!(pure_keys, key_s[i])
            push!(pure_values, value_s[i])
        end
    end
    return dicttype(pure_keys .=> pure_values)
end

@inline function timeStamp(; format = "yyyy_mm_dd_HH_MM_SS")::String
    return Dates.format(now(), format)
end

end # module Utility
