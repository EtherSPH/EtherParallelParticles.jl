#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/02/15 16:36:55
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

abstract type AbstractParticleSystemBase{IT <: Integer, FT <: AbstractFloat, Backend} end

struct ParticleSystemBase{IT <: Integer, FT <: AbstractFloat, Backend} <: AbstractParticleSystemBase{IT, FT, Backend}
    n_particles_::AbstractArray{IT, 1} # (1, ) on device, n_capacity_ >= n_particles_[1]
    is_alive_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0: dead, 1: alive
    cell_index_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0 waits for initialization
    int_properties_::AbstractArray{IT, 2} # (n_capacity, n_int_capacity) on device
    float_properties_::AbstractArray{FT, 2} # (n_capacity, n_float_capacity) on device
end

@inline function syncto!(
    destination_base::ParticleSystemBase{IT, FT, Backend1},
    source_base::ParticleSystemBase{IT, FT, Backend2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Backend1, Backend2}
    # serial copy
    Base.copyto!(destination_base.n_particles_, source_base.n_particles_)
    Base.copyto!(destination_base.is_alive_, source_base.is_alive_)
    Base.copyto!(destination_base.cell_index_, source_base.cell_index_)
    Base.copyto!(destination_base.int_properties_, source_base.int_properties_)
    Base.copyto!(destination_base.float_properties_, source_base.float_properties_)
    return nothing
end

@inline function asyncto!(
    destination_base::ParticleSystemBase{IT, FT, Backend1},
    source_base::ParticleSystemBase{IT, FT, Backend2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Backend1, Backend2}
    # async copy
    none_float_properties_task = Threads.@spawn begin
        Base.copyto!(destination_base.n_particles_, source_base.n_particles_)
        Base.copyto!(destination_base.is_alive_, source_base.is_alive_)
        Base.copyto!(destination_base.cell_index_, source_base.cell_index_)
        Base.copyto!(destination_base.int_properties_, source_base.int_properties_)
    end
    float_properties_task = Threads.@spawn begin
        Base.copyto!(destination_base.float_properties_, source_base.float_properties_)
    end
    Base.fetch(none_float_properties_task)
    Base.fetch(float_properties_task)
    return nothing
end

@inline function ParticleSystemBase(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    n_capacity::Integer,
    n_int_capacity::Integer,
    n_float_capacity::Integer,
)::ParticleSystemBase{IT, FT, Backend} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    n_int_capacity::IT = n_int_capacity
    n_float_capacity::IT = n_float_capacity
    n_capacity::IT = parallel(n_capacity)
    n_particles = parallel(IT[0])
    is_alive = parallel(zeros(IT, n_capacity)) # default: dead
    cell_index = parallel(zeros(IT, n_capacity)) # default: 0
    int_properties = parallel(zeros(IT, n_capacity, n_int_capacity))
    float_properties = parallel(zeros(FT, n_capacity, n_float_capacity))
    return ParticleSystemBase{IT, FT, Backend}(n_particles, is_alive, cell_index, int_properties, float_properties)
end

function Base.show(
    io::IO,
    particle_system_base::ParticleSystemBase{IT, FT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, Backend}
    println(io, "ParticleSystemBase{$IT, $FT, $Backend}(")
    println(io, "    n_particles: $(Array(particle_system_base.n_particles_)[1])")
    println(io, "    n_capacity: $(length(particle_system_base.is_alive_))")
    println(io, "    n_alive particles: $(sum(particle_system_base.is_alive_))")
    println(io, "    n_int_capacity: $(size(particle_system_base.int_properties_, 2))")
    println(io, "    n_float_capacity: $(size(particle_system_base.float_properties_, 2))")
    println(io, ")")
end
