#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/20 19:56:02
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Utility" begin
    @testset "Dict & NamedTuple" begin
        using OrderedCollections
        @test EtherParallelParticles.Utility.convertStringToSymbol(Dict("b" => "a", "a" => "b"); dicttype = Dict) ==
              Dict(:b => :a, :a => :b)
        @test EtherParallelParticles.Utility.convertSymbolToString(Dict(:b => :a, :a => :b); dicttype = Dict) ==
              Dict("b" => "a", "a" => "b")
        @test EtherParallelParticles.Utility.convertNamedTupleToDict((b = 1, a = 2.0); dicttype = Dict) ==
              Dict("b" => 1, "a" => 2.0)
        @test EtherParallelParticles.Utility.convertNamedTupleToDict((b = 1, a = 2.0); dicttype = OrderedDict) ==
              OrderedDict("b" => 1, "a" => 2.0)
        @test EtherParallelParticles.Utility.convertDictToNamedTuple(OrderedDict("b" => 1, "a" => 2.0)) ==
              (b = 1, a = 2.0)
        @test EtherParallelParticles.Utility.convertDictToNamedTuple(OrderedDict(:b => 1, :a => 2.0)) ==
              (b = 1, a = 2.0)
    end
end
