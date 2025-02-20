#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/20 16:54:27
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Utility

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

end # module Utility
