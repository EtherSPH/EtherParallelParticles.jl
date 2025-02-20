#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/10 17:23:54
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Parallel" begin
    if USING_CPU == true
        @testset "Parallel CPU" begin
            include("../cpu_test_head.jl")
            @test EtherParallelParticles.Environment.get_int_type(parallel) == Int32
            @test EtherParallelParticles.Environment.get_float_type(parallel) == Float32
            @test EtherParallelParticles.Environment.get_container_type(parallel) == Array
            @test EtherParallelParticles.Environment.get_backend(parallel) == KernelAbstractions.CPU()
            @test parallel(UInt8(1)) == 1
            @test parallel(1.0) ≈ 1.0f0
            @test parallel(Int64.(1:3)) == [1, 2, 3]
            @test parallel(Float64.(1:3)) ≈ [1.0f0, 2.0f0, 3.0f0]
            @kwdef struct LinearTransferCPU
                w::Int32 = 1
                b::Float32 = 2.0f0
            end
            lt = LinearTransferCPU()
            named_tuple = (a = 1, b = 2.0, lt = lt)
            @test parallel(named_tuple) == (a = IT(1), b = FT(2.0), lt = lt)
            EtherParallelParticles.Environment.synchronize(parallel)
            @test EtherParallelParticles.Environment.toHost(parallel, [1, 2, 3]) == [1, 2, 3]
            @test EtherParallelParticles.Environment.toHost(parallel, [1.0f0, 2.0f0, 3.0f0]) ≈ [1.0f0, 2.0f0, 3.0f0]
            @test parallel((mass = 1, density = 1, velocity = 2)) == (mass = IT(1), density = IT(1), velocity = IT(2))
        end
    end
    if USING_CUDA == true
        @testset "Parallel CUDA" begin
            include("../cuda_test_head.jl")
            @test EtherParallelParticles.Environment.get_int_type(parallel) == Int32
            @test EtherParallelParticles.Environment.get_float_type(parallel) == Float32
            @test EtherParallelParticles.Environment.get_container_type(parallel) == CuArray
            @test EtherParallelParticles.Environment.get_backend(parallel) == CUDA.CUDABackend()
            @test parallel(UInt8(1)) == 1
            @test parallel(1.0) ≈ 1.0f0
            @test parallel(Int64.(1:3)) |> Array == [1, 2, 3]
            @test parallel(Float64.(1:3)) |> Array ≈ [1.0f0, 2.0f0, 3.0f0]
            @kwdef struct LinearTransferCUDA
                w::Int32 = 1
                b::Float32 = 2.0f0
            end
            lt = LinearTransferCUDA()
            named_tuple = (a = 1, b = 2.0, lt = lt)
            @test parallel(named_tuple) == (a = IT(1), b = FT(2.0), lt = lt)
            EtherParallelParticles.Environment.synchronize(parallel)
            @test EtherParallelParticles.Environment.toHost(parallel, CuArray([1, 2, 3])) == [1, 2, 3]
            @test EtherParallelParticles.Environment.toHost(parallel, CuArray([1.0f0, 2.0f0, 3.0f0])) ≈
                  [1.0f0, 2.0f0, 3.0f0]
            @test parallel((mass = 1, density = 1, velocity = 2)) == (mass = IT(1), density = IT(1), velocity = IT(2))
        end
    end
    if USING_ONEAPI == true
        @testset "Parallel ONEAPI" begin
            include("../oneapi_test_head.jl")
            @test EtherParallelParticles.Environment.get_int_type(parallel) == Int32
            @test EtherParallelParticles.Environment.get_float_type(parallel) == Float32
            @test EtherParallelParticles.Environment.get_container_type(parallel) == oneArray
            @test EtherParallelParticles.Environment.get_backend(parallel) == oneAPI.oneAPIBackend()
            @test parallel(UInt8(1)) == 1
            @test parallel(1.0) ≈ 1.0f0
            @test parallel(Int64.(1:3)) |> Array == IT[1, 2, 3]
            @test parallel(Float64.(1:3)) |> Array ≈ [1.0f0, 2.0f0, 3.0f0]
            @kwdef struct LinearTransferONEAPI
                w::Int32 = 1
                b::Float32 = 2.0f0
            end
            lt = LinearTransferONEAPI()
            named_tuple = (a = 1, b = 2.0, lt = lt)
            @test parallel(named_tuple) == (a = IT(1), b = FT(2.0), lt = lt)
            EtherParallelParticles.Environment.synchronize(parallel)
            @test EtherParallelParticles.Environment.toHost(parallel, oneArray([1, 2, 3])) == [1, 2, 3]
            @test EtherParallelParticles.Environment.toHost(parallel, oneArray([1.0f0, 2.0f0, 3.0f0])) ≈
                  [1.0f0, 2.0f0, 3.0f0]
            @test parallel((mass = 1, density = 1, velocity = 2)) == (mass = IT(1), density = IT(1), velocity = IT(2))
        end
    end
end
