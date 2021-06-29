using ClimateMachine
ClimateMachine.init()

using ClimateMachine.Atmos

using ClimateMachine.ConfigTypes

using ClimateMachine.Diagnostics

using ClimateMachine.GenericCallbacks

using ClimateMachine.ODESolvers

using ClimateMachine.Mesh.Filters

using ClimateMachine.TemperatureProfiles

using ClimateMachine.MoistThermodynamics

using ClimateMachine.VariableTemplates

using StaticArrays

using Test

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_risingbubble!(bl, state, aux, (x, y, z), t)

    FT = eltype(state)

    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    xc::FT = 1250
    yc::FT = 1250
    zc::FT = 1000
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    rc::FT = 500

    θ_ref::FT = bl.ref_state.virtual_temperature_profile.T_surface
    Δθ::FT = 0

    if r <= rc
        Δθ = FT(5) * cospi(r / rc / 2)
    end

    θ = θ_ref + Δθ                                      # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      # density
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)
    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    #State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)# potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy

    ρχ = FT(0)                                          # tracer

    if 500 < z <= 550
        ρχ += FT(0.05)
    end

    ntracers = 4

    ρχ = SVector{ntracers, FT}(ρχ, ρχ / 2, ρχ / 3, ρχ / 4)

    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
    state.tracers.ρχ = ρχ
end

function config_risingbubble(FT, N, resolution, xmax, ymax, zmax)

    ode_solver = ClimateMachine.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10,
    )

    ntracers = 4
    δ_χ = SVector{ntracers, FT}(1, 2, 3, 4)

    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile)

    #md # !!! note
    #md #     Docs on model subcomponent options can be found here:
    #md #     - [`param_set`](https://CliMA.github.io/CLIMAParameters.jl/latest/)
    #md #     - `turbulence`
    #md #     - `hyperdiffusion`
    #md #     - `source`
    #md #     - `tracers`
    #md #     - `init_state`

    _C_smag = FT(C_smag(param_set))
    model = AtmosModel{FT}(
        AtmosLESConfigType,                           # Flow in a box, requires the AtmosLESConfigType
        param_set;                                    # Parameter set corresponding to earth parameters
        turbulence = SmagorinskyLilly(_C_smag),       # Turbulence closure model
        moisture = DryModel(),                        # Exclude moisture variables
        hyperdiffusion = StandardHyperDiffusion(60),  # Hyperdiffusion (4th order) model
        source = (Gravity(),),                        # Gravity is the only source term here
        tracers = NTracers{ntracers, FT}(δ_χ),        # Tracer model with diffusivity coefficients
        ref_state = ref_state,                        # Reference state
        init_state_conservative = init_risingbubble!, # Apply the initial condition
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "DryRisingBubble",       # Problem title [String]
        N,                       # Polynomial order [Int]
        resolution,              # (Δx, Δy, Δz) effective resolution [m]
        xmax,                    # Domain maximum size [m]
        ymax,                    # Domain maximum size [m]
        zmax,                    # Domain maximum size [m]
        param_set,               # Parameter set.
        init_risingbubble!,      # Function specifying initial condition
        solver_type = ode_solver,# Time-integrator type
        model = model,           # Model type
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()

    FT = Float64

    N = 4
    Δh = FT(50)
    Δv = FT(50)
    resolution = (Δh, Δh, Δv)
    xmax = FT(2500)
    ymax = FT(2500)
    zmax = FT(2500)
    t0 = FT(0)
    timeend = FT(1000)
    CFL = FT(20)

    driver_config = config_risingbubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

