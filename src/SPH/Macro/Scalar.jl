#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/04 18:26:58
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

# * use in scope with `PM` and `IP`
const kIntScalarDict = Dict("Tag" => ["tag"], "nCount" => ["count"])
# * use in scope with `PM` and `FP`
const kFloatScalarDict = Dict(
    "Mass" => ["mass", "m"],
    "Volume" => ["vol", "volume"],
    "Density" => ["rho", "ρ", "density"],
    "dDensity" => ["drho", "dρ", "density_ratio"],
    "Pressure" => ["p", "pressure", "P"],
    "Gap" => ["gap", "Δx", "Δp"],
    "H" => ["h"],
    "SumWeight" => ["wv", "∑wv"], # ∑wᵢⱼVⱼ
    "SumWeightedDensity" => ["wv_rho", "∑wv_rho", "wv_ρ", "∑wv_ρ"], # ∑wᵢⱼVⱼρⱼ
    "SumWeightedPressure" => ["wv_p", "∑wv_p"], # ∑wᵢⱼVⱼPⱼ
)
# * use in scope with `PM` and `NI` and `IP`
const kNeighbourIntScalarDict = Dict("nIndex" => ["n_index"])
# * use in scope with `PM` and `NI` and `FP`
const kNeighbourFloatScalarDict = Dict(
    "nR" => ["r", "n_r"],
    "nW" => ["w", "n_w"],
    "nDW" => ["dw", "∇w", "n_dw", "n_∇w"],
    "nHInv" => ["hinv", "n_h_inv", "h_inv"],
)

# """
# eg. 
# - `@Tag` return `PM.Tag`
# - `@Tag(i)` return `IP[i, PM.Tag]`
# """
for key in keys(kIntScalarDict)
    names = kIntScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(getfield(PM, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(i)
            return esc(:(IP[\$i, PM.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@Mass` return `PM.Mass`
# - `@Mass(i)` return `FP[i, PM.Mass]`
# """
for key in keys(kFloatScalarDict)
    names = kFloatScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(getfield(PM, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(i)
            return esc(:(FP[\$i, PM.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@nIndex` return `NI + PM.nIndex`
# - `@nIndex(ni)` return `IP[I, ni + PM.nIndex]` which is `J`
# """
for key in keys(kNeighbourIntScalarDict)
    names = kNeighbourIntScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(NI + getfield(PM, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(ni)
            return esc(:(IP[I, \$ni + PM.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@nR` return `NI + PM.nR`
# - `@nR(ni)` return `FP[I, ni + PM.nR]`
# """
for key in keys(kNeighbourFloatScalarDict)
    names = kNeighbourFloatScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(NI + getfield(PM, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(ni)
            return esc(:(FP[I, \$ni + PM.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end
