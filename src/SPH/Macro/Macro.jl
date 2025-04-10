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
although @BigCamelCase is provided, the macro-style like @snake_case is highly recommended 
"""

macro i()
    return esc(:(I))
end

macro j()
    return esc(:(IP[I, PM.nIndex + NI]))
end

macro ij()
    return esc(:(NI))
end

macro self_args()
    return esc(:((Dimension, I, IP, FP, PM)...))
end

macro inter_args()
    return esc(:((Dimension, I, NI, IP, FP, PM)...))
end

macro ci()
    return esc(:(I))
end

macro cj()
    return esc(:(J))
end

macro criterion_args()
    return esc(:((domain, I, J, IP, FP, PM, dr_square)...))
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

export @i, @j, @ij
export @self_args, @inter_args
export @int, @float
export @ci, @cj
export @criterion_args

include("Scalar.jl")
include("Vector.jl")

end # module Macro
