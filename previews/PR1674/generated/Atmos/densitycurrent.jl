using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations

using ClimateMachine.ConfigTypes

using ClimateMachine.Diagnostics

using ClimateMachine.GenericCallbacks

using ClimateMachine.ODESolvers

using ClimateMachine.Mesh.Filters

using ClimateMachine.TemperatureProfiles

using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures

using ClimateMachine.VariableTemplates

using StaticArrays

using Test

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_densitycurrent!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    # Problem float-type
    FT = eltype(state)

    # Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    # Define bubble center and background potential temperature
    xc::FT = 0
    yc::FT = 0
    zc::FT = 3000
    rx::FT = 4000
    rz::FT = 2000
    r = sqrt(((x - xc)^2) / rx^2 + ((z - zc)^2) / rz^2)

    # TODO: clean this up, or add convenience function:
    # This is configured in the reference hydrostatic state
    θ_ref::FT = bl.ref_state.virtual_temperature_profile.T_surface
    Δθ::FT = 0
    θamplitude::FT = -15.0

    # Compute temperature difference over bubble region
    if r <= 1
        Δθ = 0.5 * θamplitude * (1 + cospi(r))
    end

    # Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      ## potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z             ## exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      ## density
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)
    ρu = SVector(FT(0), FT(0), FT(0))                   ## momentum
    # State (prognostic) variable assignment
    e_kin = FT(0)                                       ## kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)## potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         ## total energy

    # Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
end

function config_densitycurrent(FT, N, resolution, xmax, ymax, zmax)

    # Choose an Explicit Single-rate Solver LSRK144 from the existing [ODESolvers](@ref
    # ODESolvers-docs) options Apply the outer constructor to define the
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    # The model coefficient for the turbulence closure is defined via the
    # [CLIMAParameters
    # package](https://CliMA.github.io/CLIMAParameters.jl/dev/) A reference
    # state for the linearisation step is also defined.
    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile)

    # The fun part! Here we assemble the `AtmosModel`.
    ##md # !!! note
    ##md #     Docs on model subcomponent options can be found here:
    ##md #     - [`param_set`](https://CliMA.github.io/CLIMAParameters.jl/dev/)
    ##md #     - `turbulence`
    ##md #     - `source`
    ##md #     - `init_state`

    _C_smag = FT(0.21)
    model = AtmosModel{FT}(
        AtmosLESConfigType,                             # Flow in a box, requires the AtmosLESConfigType
        param_set;                                      # Parameter set corresponding to earth parameters
        init_state_prognostic = init_densitycurrent!,   # Apply the initial condition
        ref_state = ref_state,                          # Reference state
        turbulence = Vreman(_C_smag),                   # Turbulence closure model
        moisture = DryModel(),                          # Exclude moisture variables
        source = (Gravity(),),                          # Gravity is the only source term here
        tracers = NoTracers(),                          # Tracer model with diffusivity coefficients
    )

    # Finally, we pass a `Problem Name` string, the mesh information, and the
    # model type to  the `AtmosLESConfiguration` object.
    config = ClimateMachine.AtmosLESConfiguration(
        "DryDensitycurrent",      # Problem title [String]
        N,                        # Polynomial order [Int]
        resolution,               # (Δx, Δy, Δz) effective resolution [m]
        xmax,                     # Domain maximum size [m]
        ymax,                     # Domain maximum size [m]
        zmax,                     # Domain maximum size [m]
        param_set,                # Parameter set.
        init_densitycurrent!,     # Function specifying initial condition
        solver_type = ode_solver, # Time-integrator type
        model = model,            # Model type
        periodicity = (false, false, false),
        boundary = ((1, 1), (1, 1), (1, 1)),   # Set all boundaries to solid walls
    )
    return config
end


function main()
    # These are essentially arguments passed to the
    # `config_densitycurrent` function.  For type
    # consistency we explicitly define the problem floating-precision.
    FT = Float64
    # We need to specify the polynomial order for the DG discretization,
    # effective resolution, simulation end-time, the domain bounds, and the
    # courant-number for the time-integrator. Note how the time-integration
    # components `solver_config` are distinct from the spatial / model
    # components in `driver_config`. `init_on_cpu` is a helper keyword argument
    # that forces problem initialisation on CPU (thereby allowing the use of
    # random seeds, spline interpolants and other special functions at the
    # initialisation step.)
    N = 4
    Δx = FT(100)
    Δy = FT(250)
    Δv = FT(100)
    resolution = (Δx, Δy, Δv)
    xmax = FT(25600)
    ymax = FT(1000)
    zmax = FT(6400)
    t0 = FT(0)
    timeend = FT(100)
    CFL = FT(1.5)

    # Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_densitycurrent(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )

    # Invoke solver (calls `solve!` function for time-integrator), pass the driver, solver and diagnostic config
    # information.
    result =
        ClimateMachine.invoke!(solver_config; check_euclidean_distance = true)

    # Check that the solution norm is reasonable.
    @test isapprox(result, FT(1); atol = 1.5e-2)
end

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

