using Test, MPI
using Random
using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.BalanceLaws
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using Thermodynamics.TemperatureProfiles
using Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.Mesh.Grids: _x1, _x2, _x3
import ClimateMachine.BalanceLaws: vars_state
import ClimateMachine.VariableTemplates.varsindex

# ------------------------ Description ------------------------- #
# 1) Dry Rising Bubble (circular potential temperature perturbation)
# 2) Boundaries - `All Walls` : Impenetrable(FreeSlip())
#                               Laterally periodic
# 3) Domain - 2500m[horizontal] x 2500m[horizontal] x 2500m[vertical]
# 4) Timeend - 1000s
# 5) Mesh Aspect Ratio (Effective resolution) 1:1
# 7) Overrides defaults for
#               `init_on_cpu`
#               `solver_type`
#               `sources`
#               `C_smag`
# 8) Default settings can be found in `src/Driver/Configurations.jl`
# ------------------------ Description ------------------------- #
function init_risingbubble!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    FT = eltype(state)
    param_set = parameter_set(bl)
    R_gas::FT = R_d(param_set)
    c_p::FT = cp_d(param_set)
    c_v::FT = cv_d(param_set)
    γ::FT = c_p / c_v
    p0::FT = MSLP(param_set)
    _grav::FT = grav(param_set)

    xc::FT = 1250
    yc::FT = 1250
    zc::FT = 1000
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    rc::FT = 500
    θ_ref::FT = 300
    Δθ::FT = 0

    if r <= rc
        Δθ = FT(5) * cospi(r / rc / 2)
    end

    #Perturbed state:
    θ = θ_ref + Δθ # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas) # density
    q_tot = FT(0)
    ts = PhaseEquil_ρθq(param_set, ρ, θ, q_tot)
    q_pt = PhasePartition(ts)

    ρu = SVector(FT(0), FT(0), FT(0))

    #State (prognostic) variable assignment
    e_kin = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)
    state.ρ = ρ
    state.ρu = ρu
    state.energy.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_pt.tot
end

function config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

    # Set up the model
    T_profile = DryAdiabaticProfile{FT}(param_set)
    C_smag = FT(0.23)
    ref_state = HydrostaticState(T_profile)
    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = SmagorinskyLilly{FT}(C_smag),
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        physics;
        init_state_prognostic = init_risingbubble!,
        source = (Gravity(),),
    )

    # Problem configuration
    config = ClimateMachine.AtmosLESConfiguration(
        "DryRisingBubble",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_risingbubble!,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end
#-------------------------------------------------------------------------
function run_brick_diagostics_fields_test()
    DA = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD
    root = 0
    pid = MPI.Comm_rank(mpicomm)
    npr = MPI.Comm_size(mpicomm)
    toler = Dict(Float64 => 1e-8, Float32 => 1e-4)
    # Working precision
    for FT in (Float64,)#(Float32, Float64)
        # DG polynomial order
        N = (4, 5)
        # Domain resolution and size
        Δh = FT(50)
        Δv = FT(50)
        resolution = (Δh, Δh, Δv)
        # Domain extents
        xmax = FT(2500)
        ymax = FT(2500)
        zmax = FT(2500)
        # Simulation time
        t0 = FT(0)
        timeend = FT(1000)

        # Courant number
        CFL = FT(20)

        driver_config = config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

        # Choose explicit multirate solver
        ode_solver_type = ClimateMachine.MultirateSolverType(
            fast_model = AtmosAcousticGravityLinearModel,
            slow_method = LSRK144NiegemannDiehlBusch,
            fast_method = LSRK144NiegemannDiehlBusch,
            timestep_ratio = 10,
        )

        solver_config = ClimateMachine.SolverConfiguration(
            t0,
            timeend,
            driver_config;
            ode_solver_type = ode_solver_type,
            init_on_cpu = true,
            Courant_number = CFL,
        )
        #------------------------------------------------------------------------

        model = driver_config.bl
        Q = solver_config.Q
        dg = solver_config.dg
        grid = dg.grid
        vgeo = grid.vgeo
        Nel = size(Q.realdata, 3)
        @show Nel
        Npl = size(Q.realdata, 1)

        ind = [
            varsindex(vars_state(model, Prognostic(), FT), :ρ)
            varsindex(vars_state(model, Prognostic(), FT), :ρu)
        ]
        _ρ, _ρu, _ρv, _ρw = ind[1], ind[2], ind[3], ind[4]


        x1 = view(vgeo, :, _x1, 1:Nel)
        x2 = view(vgeo, :, _x2, 1:Nel)
        x3 = view(vgeo, :, _x3, 1:Nel)

        fcn0(x, y, z, xmax, ymax, zmax) =
            sin.(pi * x ./ xmax) .* cos.(pi * y ./ ymax) .* cos.(pi * z ./ zmax)    # sample function

        fcnx(x, y, z, xmax, ymax, zmax) =
            cos.(pi * x ./ xmax) .* cos.(pi * y ./ ymax) .*
            cos.(pi * z ./ zmax) .* pi ./ xmax # ∂/∂x
        fcny(x, y, z, xmax, ymax, zmax) =
            -sin.(pi * x ./ xmax) .* sin.(pi * y ./ ymax) .*
            cos.(pi * z ./ zmax) .* pi ./ ymax # ∂/∂y
        fcnz(x, y, z, xmax, ymax, zmax) =
            -sin.(pi * x ./ xmax) .* cos.(pi * y ./ ymax) .*
            sin.(pi * z ./ zmax) .* pi ./ zmax # ∂/∂z

        Q.data[:, _ρ, 1:Nel] .= 1.0 .+ fcn0(x1, x2, x3, xmax, ymax, zmax) * 5.0
        Q.data[:, _ρu, 1:Nel] .=
            Q.data[:, _ρ, 1:Nel] .* fcn0(x1, x2, x3, xmax, ymax, zmax)
        Q.data[:, _ρv, 1:Nel] .=
            Q.data[:, _ρ, 1:Nel] .* fcn0(x1, x2, x3, xmax, ymax, zmax)
        Q.data[:, _ρw, 1:Nel] .=
            Q.data[:, _ρ, 1:Nel] .* fcn0(x1, x2, x3, xmax, ymax, zmax)
        #-----------------------------------------------------------------------
        vgrad = Diagnostics.VectorGradients(dg, Q)
        vort = Diagnostics.Vorticity(dg, vgrad)
        #----------------------------------------------------------------------------
        Ω₁_exact =
            fcny(x1, x2, x3, xmax, ymax, zmax) -
            fcnz(x1, x2, x3, xmax, ymax, zmax)
        Ω₂_exact =
            fcnz(x1, x2, x3, xmax, ymax, zmax) -
            fcnx(x1, x2, x3, xmax, ymax, zmax)
        Ω₃_exact =
            fcnx(x1, x2, x3, xmax, ymax, zmax) -
            fcny(x1, x2, x3, xmax, ymax, zmax)

        err = zeros(FT, 12)

        err[1] = maximum(abs.(fcnx(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₁u₁))
        err[2] = maximum(abs.(fcny(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₂u₁))
        err[3] = maximum(abs.(fcnz(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₃u₁))

        err[4] = maximum(abs.(fcnx(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₁u₂))
        err[5] = maximum(abs.(fcny(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₂u₂))
        err[6] = maximum(abs.(fcnz(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₃u₂))

        err[7] = maximum(abs.(fcnx(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₁u₃))
        err[8] = maximum(abs.(fcny(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₂u₃))
        err[9] = maximum(abs.(fcnz(x1, x2, x3, xmax, ymax, zmax) - vgrad.∂₃u₃))

        err[10] = maximum(abs.(vort.Ω₁ - Ω₁_exact))
        err[11] = maximum(abs.(vort.Ω₂ - Ω₂_exact))
        err[12] = maximum(abs.(vort.Ω₃ - Ω₃_exact))

        errg = MPI.Allreduce(err, max, mpicomm)
        @test maximum(errg) < toler[FT]
    end
end
#----------------------------------------------------------------------------
@testset "Diagnostics Fields tests" begin
    run_brick_diagostics_fields_test()
end
#------------------------------------------------
