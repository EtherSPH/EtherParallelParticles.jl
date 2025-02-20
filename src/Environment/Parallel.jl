#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 19:57:42
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

const kCPUContainerType = Array
const kCPUBackend = KernelAbstractions.CPU()

abstract type AbstractParallel{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} end

struct Parallel{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} <:
       AbstractParallel{IT, FT, CT, Backend} end

function Base.show(
    io::IO,
    ::Parallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    println(io, "Parallel{$IT, $FT, $CT, $Backend}")
end

@inline function get_int_type(
    ::AbstractParallel{IT, FT, CT, Backend},
)::DataType where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return IT
end

@inline function get_float_type(
    ::AbstractParallel{IT, FT, CT, Backend},
)::DataType where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return FT
end

@inline function get_container_type(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return CT
end

@inline function get_backend(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return Backend
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::IntType,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, IntType <: Integer}
    return IT(x)
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::FloatType,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, FloatType <: AbstractFloat}
    return FT(x)
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::AT,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, AT <: AbstractArray{<:Integer}}
    if CT === AT
        if IT === eltype(x)
            return x
        else
            return IT.(x)
        end
    else
        return CT(IT.(x))
    end
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::AT,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, AT <: AbstractArray{<:AbstractFloat}}
    if CT === AT
        if FT === eltype(x)
            return x
        else
            return FT.(x)
        end
    else
        return CT(FT.(x))
    end
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::NamedTuple,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    name_s = fieldnames(typeof(x))
    value_s = values(x)
    converted_value_s = map(value_s) do item
        if typeof(item) <: Real
            return parallel(item)
        else
            return item
        end
    end
    return NamedTuple{name_s}(converted_value_s)
end

@inline function synchronize(
    ::AbstractParallel{IT, FT, CT, Backend},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function toDevice(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    x::Array,
)::CT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return parallel(x)
end

@inline function toDevice(
    ::AbstractParallel{IT, FT, kCPUContainerType, kCPUBackend},
    x::Array,
)::Array where {IT <: Integer, FT <: AbstractFloat}
    return deepcopy(x)
end

@inline function toHost(
    ::AbstractParallel{IT, FT, CT, Backend},
    x::CT,
)::Array where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return Array(x)
end

@inline function toHost(
    ::AbstractParallel{IT, FT, kCPUContainerType, kCPUBackend},
    x::Array,
)::Array where {IT <: Integer, FT <: AbstractFloat}
    return deepcopy(x)
end
