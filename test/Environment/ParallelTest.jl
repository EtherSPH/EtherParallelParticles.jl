#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/10 17:23:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Parallel $DEVICE" begin
    @kwdef struct LinearTransfer
        w::IT = 1
        b::FT = 2.0f0
    end
    lt = LinearTransfer()
    @test EtherParallelParticles.Environment.get_int_type(parallel) == IT
    @test EtherParallelParticles.Environment.get_float_type(parallel) == FT
    @test EtherParallelParticles.Environment.get_container_type(parallel) == CT
    @test EtherParallelParticles.Environment.get_backend(parallel) == Backend
    @test parallel(UInt8(1)) == IT(1)
    @test parallel(1.0) ≈ 1.0f0
    @test parallel(Int64.(1:3)) |> Array == [1, 2, 3]
    @test parallel(Float64.(1:3)) |> Array ≈ [1.0f0, 2.0f0, 3.0f0]
    lt = LinearTransfer()
    named_tuple = (a = 1, b = 2.0, lt = lt)
    @test parallel(named_tuple) == (a = IT(1), b = FT(2.0), lt = lt)
    EtherParallelParticles.Environment.synchronize(parallel)
    @test EtherParallelParticles.Environment.toHost(parallel, CT([1, 2, 3])) == IT[1, 2, 3]
    @test EtherParallelParticles.Environment.toHost(parallel, CT([1.0f0, 2.0f0, 3.0f0])) ≈ [1.0f0, 2.0f0, 3.0f0]
    @test parallel((mass = 1, density = 1, velocity = 2)) == (mass = IT(1), density = IT(1), velocity = IT(2))
end
