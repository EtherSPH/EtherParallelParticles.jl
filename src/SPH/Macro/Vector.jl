#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/06 16:22:49
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

# * use in scope with `PM` and `FP`
const kFloatVectorDict = Dict(
    "PositionVec" => ["x", "Position", "position"],
    "VelocityVec" => ["u", "Velocity", "velocity"],
    "dVelocityVec" => ["du", "dVelocity", "dvelocity"],
    "AccelerationVec" => ["a", "Acceleration", "acceleration"],
)
# * use in scope with `PM` and `NI` and `FP`
const kNeighbourFloatVectorDict = Dict("nRVec" => ["n_rvec", "rvec"])

# """
# eg.
# - `@VelocityVec` return `PM.VelocityVec`
# - `@VelocityVec(index, i)` return `FP[index, PM.VelocityVec + i]`
# """
for key in keys(kFloatVectorDict)
    names = kFloatVectorDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(getfield(PM, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(index, i)
            return esc(:(FP[\$index, PM.$key + \$i]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@nRVec` return `NI + PM.nRVec`
# - `@nR(ni, i)` return `FP[I, ni + PM.nR + i]`
# """
for key in keys(kNeighbourFloatVectorDict)
    names = kNeighbourFloatVectorDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(NI + getfield(PM, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(ni, i)
            return esc(:(FP[I, \$ni + PM.$key + \$i]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end
