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
    d_particle_system = EtherParallelParticles.Class.ParticleSystem(
        parallel,
        domain,
        n_particles,
        int_named_tuple,
        float_named_tuple,
        parameters;
        capacityExpand = capacityExpand,
    )
    h_particle_system = EtherParallelParticles.Class.mirror(d_particle_system)
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
        h_particle_system.base_.float_properties_[i, h_particle_system.basic_index_.PositionVec] = xy[i, 1]
        h_particle_system.base_.float_properties_[i, h_particle_system.basic_index_.PositionVec + 1] = xy[i, 2]
        h_particle_system.base_.is_alive_[i] = 1
        h_particle_system.base_.int_properties_[i, h_particle_system.basic_index_.Tag] = 1
    end
    EtherParallelParticles.Class.asyncto!(d_particle_system, h_particle_system)
    EtherParallelParticles.Algorithm.search!(d_particle_system, domain, neighbour_system)
    kernel = SPH.Kernel.CubicSpline{IT, FT, dim}()
    h_inv = 1 / (FT(1.5) * gap)
    @inline function inter!(Dimension, I, NI, IP, FP, PM)
        w = SPH.Library.iValue(Dimension, I, NI, IP, FP, PM, kernel)
        dw = SPH.Library.iGradient(Dimension, I, NI, IP, FP, PM, kernel)
        SPH.Library.iClassicContinuity!(Dimension, I, NI, IP, FP, PM; dw = dw)
        SPH.Library.iBalancedContinuity!(Dimension, I, NI, IP, FP, PM; dw = dw)
        SPH.Library.iValue!(Dimension, I, NI, IP, FP, PM, kernel)
        SPH.Library.iGradient!(Dimension, I, NI, IP, FP, PM, kernel)
        SPH.Library.iValue!(Dimension, I, NI, IP, FP, PM, h_inv, kernel)
        SPH.Library.iGradient!(Dimension, I, NI, IP, FP, PM, h_inv, kernel)
        SPH.Library.iValueGradient!(Dimension, I, NI, IP, FP, PM, kernel)
        p_c = SPH.Library.iPressureCorrection(Dimension, I, NI, IP, FP, PM, kernel; hinv = h_inv, w = w)
        SPH.Library.iClassicPressure!(Dimension, I, NI, IP, FP, PM; dw = dw, coefficient = p_c)
        SPH.Library.iBalancedPressure!(Dimension, I, NI, IP, FP, PM; dw = dw, coefficient = p_c)
        SPH.Library.iDensityWeightedPressure!(Dimension, I, NI, IP, FP, PM; dw = dw, coefficient = p_c)
        SPH.Library.iExtrapolatePressure!(Dimension, I, NI, IP, FP, PM; w = w, p0 = 0.0, gx = 0.0, gy = -9.8)
        SPH.Library.iClassicViscosity!(Dimension, I, NI, IP, FP, PM; dw = dw, mu = 1e-3)
        SPH.Library.sArtificialViscosity!(Dimension, I, NI, IP, FP, PM; dw = dw, alpha = 0.1, beta = 0.1, c = PM.c_0)
        SPH.Library.iKernelFilter!(Dimension, I, NI, IP, FP, PM; w = w)
    end
    @inline function self!(Dimension, I, IP, FP, PM)
        w0 = SPH.Kernel.value0(@h(@i), kernel)
        SPH.Library.sVolume!(Dimension, I, IP, FP, PM)
        SPH.Library.sGravity!(Dimension, I, IP, FP, PM; gx = 0, gy = -9.8)
        SPH.Library.sContinuity!(Dimension, I, IP, FP, PM; dt = 0.1)
        SPH.Library.sAcceleration!(Dimension, I, IP, FP, PM)
        SPH.Library.sAccelerate!(Dimension, I, IP, FP, PM; dt = 0.1)
        SPH.Library.sMove!(Dimension, I, IP, FP, PM; dt = 0.1)
        SPH.Library.sAccelerateMove!(Dimension, I, IP, FP, PM; dt = 0.1)
        SPH.Library.sExtrapolatePressure!(Dimension, I, IP, FP, PM; p0 = 0.0)
        SPH.Library.sKernelFilter!(Dimension, I, IP, FP, PM; w0 = w0)
    end
    Algorithm.selfaction!(d_particle_system, self!)
    Algorithm.interaction!(d_particle_system, inter!)
end
