#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/09 21:03:19
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Geometry" begin
    @testset "Geometry 2D" begin
        rectangle = Geometry.Rectangle(0.0, 0.0, 2.0, 3.0)
        @test Geometry.count(0.1, rectangle) == 600
        @test Geometry.inside(0.1, 0.2, rectangle) == true
        @test Geometry.inside(2.1, 0.2, rectangle) == false

        ring = Geometry.Ring(0.0, 0.0, 1.0, 2.0)
        @test Geometry.count(0.1, ring) == sum([round(Int, 2 * pi * r / 0.1) for r in 1.05:0.1:1.95])
        @test Geometry.inside(1.3 * cosd(60), 1.3 * sind(60), ring) == true
        @test Geometry.inside(2.1 * cosd(60), 2.1 * sind(60), ring) == false

        circle = Geometry.Circle(0.0, 0.0, 1.05)
        @test Geometry.count(0.1, circle) == 1 + sum([round(Int, 2 * pi * r / 0.1) for r in 0.1:0.1:1.0])
        @test Geometry.inside(0.5 * cosd(60), 0.5 * sind(60), circle) == true
        @test Geometry.inside(1.1 * cosd(60), 1.1 * sind(60), circle) == false
    end
end
