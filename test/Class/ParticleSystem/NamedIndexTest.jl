#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/18 15:59:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "NamedIndex" begin
    IT = Int32
    dim = 2
    neighbour_count = 50
    int_named_tuple = (Tag = 1, nCount = 1, nIndex = 1 * neighbour_count)
    float_named_tuple = (
        RVec = dim,
        Mass = 1,
        Density = 1,
        Volume = 1,
        VelocityVec = dim,
        dVelocityVec = dim,
        dDensity = 1,
        Pressure = 1,
        StrainMat = dim * dim,
        dStrainMat = dim * dim,
        StressMat = dim * dim,
        nRVec = dim * neighbour_count,
        nR = neighbour_count,
        nW = neighbour_count,
        nDW = neighbour_count,
    )
    named_index = EtherParallelParticles.Class.NamedIndex{IT}(int_named_tuple, float_named_tuple)
    @test EtherParallelParticles.Class.get_n_int_capacity(named_index) == 1 + 1 + 1 * neighbour_count
    @test EtherParallelParticles.Class.get_n_float_capacity(named_index) ==
          dim +
          1 +
          1 +
          1 +
          dim +
          dim +
          1 +
          1 +
          dim * dim +
          dim * dim +
          dim * dim +
          dim * neighbour_count +
          neighbour_count +
          neighbour_count +
          neighbour_count
end
