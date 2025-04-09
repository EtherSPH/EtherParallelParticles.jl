#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/08 17:03:51
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

using JSON
using Dates
using OrderedCollections
using EtherParallelParticles
using CUDA
using KernelAbstractions
using ProgressMeter
using EtherParallelParticles.SPH.Macro

config_dict = JSON.parsefile("example/result/collapse_dry/same/config/config.json"; dicttype = OrderedDict)
# * cpu
# config_dict["parallel"]["backend"] = "cpu"
# config_dict["parallel"]["float"] = "Float64"
# const n_threads = Threads.nthreads() - 2 # 2 threads free for output tasks
# * cuda
config_dict["parallel"]["backend"] = "cuda"
const n_threads = 1024
# * rocm
# config_dict["parallel"]["backend"] = "rocm"
# const n_threads = 1024
# * oneapi
# config_dict["parallel"]["backend"] = "oneapi"
# const n_threads = 256

const backend = config_dict["parallel"]["backend"]
const color = Environment.kNameToColor[backend]

eval(DataIO.Parallel(config_dict))

const domain = DataIO.Domain(config_dict, parallel)
const IT = typeof(domain).parameters[1]
const FT = typeof(domain).parameters[2]
const dimension = Class.dimension(domain)

device_particle_system = DataIO.ParticleSystem(config_dict, parallel, domain)
host_particle_system = Class.mirror(device_particle_system)

neighbour_system = DataIO.NeighbourSystem(config_dict, parallel, domain)
writer = DataIO.Writer(config_dict)

const FLUID_TAG = host_particle_system.parameters_.FLUID_TAG
const WALL_TAG = host_particle_system.parameters_.WALL_TAG
const rho0 = host_particle_system.parameters_.rho0
const p0 = host_particle_system.parameters_.p0
const gap0 = host_particle_system.parameters_.gap0
const h0 = host_particle_system.parameters_.h0
const mu0 = host_particle_system.parameters_.mu0
const mu0_2 = mu0 * 2
const gx = parallel(0.0)
const gy = parallel(-9.8)
const c0 = parallel(120.0)
const c02 = c0 * c0
const sph_kernel = SPH.Kernel.CubicSpline{IT, FT, Int64(dimension)}()

@inline eos(rho::Real) = c02 * (rho - rho0) + p0

const total_time = parallel(3.0)
const total_time_inv = parallel(1 / total_time)
const dt = parallel(0.1 * h0 / c0) * 2
const output_interval = 400
const filter_interval = 20

# * ===================== particle action definition ===================== * #

@inline function sComputeKernelAndiContinuity!(@self_args)::Nothing
    NI::@int() = @int 0
    @inbounds while NI < @count(@i)
        @inbounds if @tag(@i) == FLUID_TAG && @tag(@j) == FLUID_TAG
            SPH.Library.iValueGradient!(@inter_args, sph_kernel)
            SPH.Library.iClassicContinuity!(@inter_args; dw = @dw(@ij))
        elseif @tag(@i) == FLUID_TAG && @tag(@j) == WALL_TAG
            SPH.Library.iValueGradient!(@inter_args, sph_kernel)
            SPH.Library.iBalancedContinuity!(@inter_args; dw = @dw(@ij))
        elseif @tag(@i) == WALL_TAG && @tag(@j) == FLUID_TAG
            SPH.Library.iValueGradient!(@inter_args, sph_kernel)
            SPH.Library.iBalancedContinuity!(@inter_args; dw = @dw(@ij))
        end
        NI += 1
    end
    return nothing
end

@inline function sContinuity!(@self_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG
        SPH.Library.sContinuity!(@self_args; dt = dt)
        SPH.Library.sVolume!(@self_args)
        @inbounds @p(@i) = eos(@rho(@i))
        return nothing
    elseif @tag(@i) == WALL_TAG
        SPH.Library.sContinuity!(@self_args; dt = dt)
        SPH.Library.sVolume!(@self_args)
        @inbounds @p(@i) = eos(@rho(@i))
        return nothing
    end
    return nothing
end

@inline function iMomentum!(@inter_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG && @tag(@j) == FLUID_TAG
        SPH.Library.iClassicPressure!(@inter_args; dw = @dw(@ij))
        SPH.Library.iClassicViscosity!(@inter_args; dw = @dw(@ij), mu = mu0)
        return nothing
    elseif @tag(@i) == FLUID_TAG && @tag(@j) == WALL_TAG
        SPH.Library.iBalancedPressure!(@inter_args; dw = @dw(@ij))
        SPH.Library.iClassicViscosity!(@inter_args; dw = @dw(@ij), mu = mu0)
        return nothing
    end
    return nothing
end

@inline function sMomentum!(@self_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG
        SPH.Library.sGravity!(@self_args; gx = gx, gy = gy)
        SPH.Library.sAccelerateMove!(@self_args; dt = dt)
    end
    return nothing
end

@inline function iFilter(@inter_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG && @tag(@j) == FLUID_TAG
        SPH.Library.iKernelFilter!(@inter_args; w = @w(@ij))
        return nothing
    end
    return nothing
end

@inline function sFilter(@self_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG
        SPH.Library.sKernelFilter!(@self_args; w0 = SPH.Kernel.value0(@h(@i), sph_kernel))
        SPH.Library.sVolume!(@self_args)
        @inbounds @p(@i) = eos(@rho(@i))
        return nothing
    end
    return nothing
end

# * ===================== simulation function definition ===================== * #

function main(step = :first)
    appendix = DataIO.appendix()
    DataIO.mkdir(writer)
    DataIO.load!(writer, host_particle_system; appendix = appendix, step = step)
    t = parallel(appendix["TimeValue"])
    step = parallel(appendix["TMSTEP"])
    write_step = parallel(appendix["WriteStep"])
    percentage = t * total_time_inv
    Class.asyncto!(device_particle_system, host_particle_system)
    Algorithm.search!(
        device_particle_system,
        domain,
        neighbour_system;
        n_threads = n_threads,
        action! = sComputeKernelAndiContinuity!,
        criterion = Algorithm.symmetrySearchCriterion,
    )
    progress = ProgressMeter.ProgressThresh(0.0; desc = "Task Left:", dt = 0.1, color = color, showspeed = true)
    while t <= total_time
        t += dt
        step += 1
        Algorithm.selfaction!(device_particle_system, sContinuity!; n_threads = n_threads)
        Algorithm.interaction!(device_particle_system, iMomentum!; n_threads = n_threads)
        Algorithm.selfaction!(device_particle_system, sMomentum!; n_threads = n_threads)
        Algorithm.search!(
            device_particle_system,
            domain,
            neighbour_system;
            n_threads = n_threads,
            action! = sComputeKernelAndiContinuity!,
            criterion = Algorithm.symmetrySearchCriterion,
        )
        if step % filter_interval == 0
            Algorithm.interaction!(device_particle_system, iFilter; n_threads = n_threads)
            Algorithm.selfaction!(device_particle_system, sFilter; n_threads = n_threads)
        end
        if step % output_interval == 0
            write_step += 1
            appendix["TimeValue"] = t
            appendix["TMSTEP"] = step
            appendix["WriteStep"] = write_step
            DataIO.wait!(writer)
            Class.asyncto!(host_particle_system, device_particle_system)
            DataIO.save!(writer, host_particle_system, appendix;)
        end
        percentage = t * total_time_inv
        update!(
            progress,
            1 - percentage;
            showvalues = [
                ("Backend", backend),
                ("TimeValue", t),
                ("TMSTEP", step),
                ("WriteStep", write_step),
                ("Percentage %", percentage * 100),
                ("ρ₀", device_particle_system.parameters_.rho0),
            ],
            valuecolor = color,
        )
    end
end
