#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 00:08:14
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using EtherParallelParticles
using CairoMakie

result_path = joinpath(@__DIR__, "../../result/collapse_dry/same")
writer = DataIO.Writer(result_path)

appendix = DataIO.appendix();
ps = DataIO.load(writer; appendix = appendix, step = :last);
tag = ps[:Tag]
x = ps[:PositionVec][:, 1]
y = ps[:PositionVec][:, 2]
u = ps[:VelocityVec][:, 1]
v = ps[:VelocityVec][:, 2]
vel = sqrt.(u .^ 2 .+ v .^ 2)
vel[tag .== 2] .= NaN

with_theme(theme_latexfonts()) do
    t = appendix["TimeValue"]
    t = round(t, digits = 6) |> string
    fig = Figure(size = (600, 400), fontsize = 18)
    axes = Axis(
        fig[1, 1],
        aspect = DataAspect(),
        title = L"$\sqrt{u^2 + v^2} = |\vec{u}|$ at Time %$t s",
        xlabel = L"x",
        ylabel = L"y",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
    )
    s = scatter!(axes, x, y, color = vel, colormap = :roma, markersize = 2, nan_color = :gray)
    Colorbar(fig[1, 2], s, height = 300, width = 10, label = "Velocity", ticklabelsize = 16, labelpadding = 5)
    fig
end
