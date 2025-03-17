#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/03/07 18:28:15
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParallelParticles` a parallel particle-based simulator supporting multi-backend gpu in julia.
  @ description:
 =#

@testset "Library $DEVICE" begin
    domain = EtherParallelParticles.Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
    active_pair = [1 => 1, 1 => 2, 2 => 1]
    periodic_boundary_policy = EtherParallelParticles.Class.NonePeriodicBoundaryPolicy
    neighbour_system =
        EtherParallelParticles.Class.NeighbourSystem(parallel, domain, active_pair, periodic_boundary_policy)
    dim = 2
    neighbour_count = 50
    n_particles = 9
    int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = 1 * neighbour_count)
    float_named_tuple = (
        PositionVec = dim,
        nRVec = dim * neighbour_count,
        nR = neighbour_count,
        VelocityVec = dim,
        dVelocityVec = dim,
        AccelerationVec = dim,
        Mass = 1,
        Density = 1,
        dDensity = 1,
        Volume = 1,
        Pressure = 1,
        Gap = 1,
        H = 1,
        SumWeight = 1,
        SumWeightedDensity = 1,
        SumWeightedPressure = 1,
        nW = neighbour_count,
        nDW = neighbour_count,
        nHInv = neighbour_count,
    )
    parameters = (c_0 = 340.0, gamma = 7, mu = 1e-3)
    capacityExpand(n)::typeof(n) = n
    particle_system = EtherParallelParticles.Class.ParticleSystem(
        parallel,
        domain,
        n_particles,
        int_named_tuple,
        float_named_tuple,
        parameters;
        capacityExpand = capacityExpand,
    )
    start_x = EtherParallelParticles.Class.get_first_x(domain)
    start_y = EtherParallelParticles.Class.get_first_y(domain)
    last_x = EtherParallelParticles.Class.get_last_x(domain)
    last_y = EtherParallelParticles.Class.get_last_y(domain)
    gap_x = EtherParallelParticles.Class.get_gap_x(domain)
    gap_y = EtherParallelParticles.Class.get_gap_y(domain)
    gap = EtherParallelParticles.Class.get_gap(domain)
    xy = zeros(FT, 9, 2)
    err = 1e-3
    xy[1, :] .= [start_x + err, start_y + err]
    xy[2, :] .= [start_x + gap / 2, start_y + err]
    xy[3, :] .= [start_x + gap * 3 / 2 - err, start_y + err]
    xy[4, :] .= [start_x + err, start_y + gap / 2]
    xy[5, :] .= [start_x + gap / 2, start_y + gap / 2]
    xy[6, :] .= [start_x + gap * 3 / 2 - err, start_y + gap / 2]
    xy[7, :] .= [start_x + err, start_y + gap - err]
    xy[8, :] .= [start_x + gap / 2, start_y + gap - err]
    xy[9, :] .= [start_x + gap * 3 / 2 - err, start_y + gap - err]
    for i in 1:9
        particle_system.host_base_.float_properties_[i, particle_system.basic_index_.PositionVec] = xy[i, 1]
        particle_system.host_base_.float_properties_[i, particle_system.basic_index_.PositionVec + 1] = xy[i, 2]
        particle_system.host_base_.is_alive_[i] = 1
        particle_system.host_base_.int_properties_[i, particle_system.basic_index_.Tag] = 1
    end
    EtherParallelParticles.Class.toDevice!(particle_system)
    EtherParallelParticles.Algorithm.search!(particle_system, domain, neighbour_system)
    kernel = SPH.Kernel.CubicSpline{IT, FT, dim}()
    h_inv = 1 / (FT(1.5) * gap)
    @inline function inter!(d, i, ni, ip, fp, pm)
        SPH.Library.classicContinuity!(d, i, ni, ip, fp, pm)
        SPH.Library.balancedContinuity!(d, i, ni, ip, fp, pm)
        SPH.Library.value!(d, i, ni, ip, fp, pm, kernel)
        SPH.Library.gradient!(d, i, ni, ip, fp, pm, kernel)
        SPH.Library.value!(d, i, ni, ip, fp, pm, h_inv, kernel)
        SPH.Library.gradient!(d, i, ni, ip, fp, pm, h_inv, kernel)
        SPH.Library.value_gradient!(d, i, ni, ip, fp, pm, kernel)
        p_c = SPH.Library.pressureCorrection(d, i, ni, ip, fp, pm, kernel)
        SPH.Library.classicPressure!(d, i, ni, ip, fp, pm; coefficient = p_c)
        SPH.Library.balancedPressure!(d, i, ni, ip, fp, pm; coefficient = p_c)
        SPH.Library.densityWeightedPressure!(d, i, ni, ip, fp, pm; coefficient = p_c)
        SPH.Library.extrapolatePressure!(d, i, ni, ip, fp, pm; p0 = 0.0, gx = 0.0, gy = -9.8)
        SPH.Library.classicViscosity!(d, i, ni, ip, fp, pm; mu = 1e-3)
        SPH.Library.artificialViscosity!(d, i, ni, ip, fp, pm; alpha = 0.1, beta = 0.1, c = pm.c_0)
        SPH.Library.kernelFilter!(d, i, ni, ip, fp, pm)
    end
    @inline function self!(d, i, ip, fp, pm)
        SPH.Library.volume!(d, i, ip, fp, pm)
        SPH.Library.gravity!(d, i, ip, fp, pm; gx = 0, gy = -9.8)
        SPH.Library.continuity!(d, i, ip, fp, pm; dt = 0.1)
        SPH.Library.acceleration!(d, i, ip, fp, pm)
        SPH.Library.accelerate!(d, i, ip, fp, pm; dt = 0.1)
        SPH.Library.move!(d, i, ip, fp, pm; dt = 0.1)
        SPH.Library.accelerate_move!(d, i, ip, fp, pm; dt = 0.1)
        SPH.Library.extrapolatePressure!(d, i, ip, fp, pm; p0 = 0.0)
        SPH.Library.kernelFilter!(d, i, ip, fp, pm, kernel)
    end
    Algorithm.selfaction!(particle_system, self!)
    Algorithm.interaction!(particle_system, inter!)
end
