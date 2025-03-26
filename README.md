# EtherParallelParticles.jl
A parallel particle-based simulator supporting multi-backend gpu in julia.

# Dependencies

- [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl): atomic operations
- [CSV.jl](https://github.com/JuliaData/CSV.jl): read and write csv files
- [CodecZstd.jl](https://github.com/JuliaIO/CodecZstd.jl): compress the `.jld` files in a balanced between speed and compression-ratio
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl): handle tabular data
- [Dates.jl](https://docs.julialang.org/en/v1/stdlib/Dates/#:~:text=The%20Dates%20module%20provides%20two%20types%20for%20working,respectively%3B%20both%20are%20subtypes%20of%20the%20abstract%20TimeType.): get time-stamp
- [JLD2.jl](https://github.com/JuliaIO/JLD2.jl): store and load raw data in `h5` format
- [JSON.jl](https://github.com/JuliaIO/JSON.jl): configure the simulation in `json` format
- [YAML.jl](https://github.com/JuliaData/YAML.jl): configure the simulation in `yaml` format
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl): an abstraction layer for parallel kernels, which is the core dependency of this package
- [OrderedCollections.jl](https://github.com/JuliaCollections/OrderedCollections.jl): don't break the configuration order
- [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl): show the progress of tasks

To run the code on gpu, one of the following backends is required:

- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl): NVIDIA discrete GPUs
- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl): AMD GPUs, both discrete and integrated
- [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl): Intel GPUs, both discrete and integrated
- [Metal.jl](https://github.com/JuliaGPU/Metal.jl): Apple M-series GPUs
