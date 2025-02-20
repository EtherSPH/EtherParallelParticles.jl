#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/14 15:58:51
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Domain" begin
    IT = Int32
    FT = Float32
    x_0 = -1.0
    y_0 = -2.0
    x_1 = 3.0
    y_1 = 3.0
    gap = 0.15
    domain_2d = EtherParallelParticles.Class.Domain2D{IT, FT}(gap, x_0, y_0, x_1, y_1)

    @test EtherParallelParticles.Class.dimension(domain_2d) == 2
    @test EtherParallelParticles.Class.get_gap(domain_2d) ≈ gap
    @test EtherParallelParticles.Class.get_gap_square(domain_2d) ≈ gap * gap
    @test EtherParallelParticles.Class.get_n_x(domain_2d) == 26
    @test EtherParallelParticles.Class.get_n_y(domain_2d) == 33
    @test EtherParallelParticles.Class.get_n(domain_2d) == 26 * 33
    @test EtherParallelParticles.Class.get_first_x(domain_2d) ≈ x_0
    @test EtherParallelParticles.Class.get_first_y(domain_2d) ≈ y_0
    @test EtherParallelParticles.Class.get_last_x(domain_2d) ≈ x_1
    @test EtherParallelParticles.Class.get_last_y(domain_2d) ≈ y_1
    @test EtherParallelParticles.Class.get_span_x(domain_2d) ≈ x_1 - x_0
    @test EtherParallelParticles.Class.get_span_y(domain_2d) ≈ y_1 - y_0
    @test EtherParallelParticles.Class.get_gap_x(domain_2d) ≈ (x_1 - x_0) / 26
    @test EtherParallelParticles.Class.get_gap_y(domain_2d) ≈ (y_1 - y_0) / 33
    @test EtherParallelParticles.Class.get_gap_x_inv(domain_2d) ≈ 1 / EtherParallelParticles.Class.get_gap_x(domain_2d)
    @test EtherParallelParticles.Class.get_gap_y_inv(domain_2d) ≈ 1 / EtherParallelParticles.Class.get_gap_y(domain_2d)

    @test EtherParallelParticles.Class.indexCartesianToLinear(domain_2d, IT(2), IT(3)) == 2 + (3 - 1) * 26
    @test EtherParallelParticles.Class.indexLinearToCartesian(domain_2d, IT(2 + (3 - 1) * 26)) == (IT(2), IT(3))
    @test EtherParallelParticles.Class.inside(domain_2d, 1.5f0, 0.3f0) == true
    @test EtherParallelParticles.Class.indexCartesianFromPosition(domain_2d, 1.5f0, 0.3f0) == (IT(17), IT(16))
    @test EtherParallelParticles.Class.indexLinearFromPosition(domain_2d, 1.5f0, 0.3f0) == IT(17 + (16 - 1) * 26)
end
