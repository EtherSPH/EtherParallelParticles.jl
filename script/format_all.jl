#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/04 19:50:03
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

# using Pkg
# Pkg.add("JuliaFormatter")
using JuliaFormatter

@info "JuliaFormatter: `format.jl` is formatting all files in the current directory"

JuliaFormatter.format(
    ".",
    indent = 4, # 4 spaces for each indent
    margin = 120, # 120 characters for each line
    always_for_in = true, # always use `for i in 1:10` instead of `for i = 1:10`
    whitespace_typedefs = true, # add whitespace around `<:`
    whitespace_ops_in_indices = true, # add whitespace around `+`, `-`, `*`, `/` in indices
    remove_extra_newlines = true, # remove extra newlines
    pipe_to_function_call = false, # add whitespace around `|>` in function call
    always_use_return = false, # ! don't use `return` in function, for kernel function doesn't allow `return`
    whitespace_in_kwargs = true, # add whitespace around `=` in kwargs
    trailing_comma = true, # add trailing comma
)
