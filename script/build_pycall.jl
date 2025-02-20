#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 19:50:03
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using Pkg
Pkg.add("Conda")
Pkg.add("PyCall")
ENV["PYTHON"] = ""
Pkg.build("PyCall")
