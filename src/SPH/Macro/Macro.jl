#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/03 00:50:26
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

module Macro

@doc md"""
this is a dangerous module, for meta-programming is used here, code is generated from strings
although @BigCamelCase is provided, the macro-style like @snake_case is also recommended 
"""

macro i()
    return esc(:(I))
end

macro j()
    return esc(:(IP[I, PM.nIndex + NI]))
end

macro ni()
    return esc(:(NI))
end

macro self_args()
    return esc(:((Dimension, I, IP, FP, PM)...))
end

macro inter_args()
    return esc(:((Dimension, I, NI, IP, FP, PM)...))
end

macro int()
    return esc(:(eltype(IP)))
end

macro int(x)
    return esc(:(eltype(IP)($x)))
end

macro float()
    return esc(:(eltype(FP)))
end

macro float(x)
    return esc(:(eltype(FP)($x)))
end

export @i, @j, @ni
export @self_args, @inter_args
export @int, @float

include("Scalar.jl")
include("Vector.jl")

end # module Macro
