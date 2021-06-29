using Distributions
using Random
using StaticArrays
using UnPack

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.TemperatureProfiles
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws

import ClimateMachine.BalanceLaws: source
import ClimateMachine.Atmos: filter_source, atmos_source!

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, cp_d, cv_d, planet_radius

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();

"""
    HeldSuarezForcingTutorial{PV <: Union{Momentum,Energy}} <: TendencyDef{Source, PV}

Defines a forcing that parametrises radiative and frictional effects using
Newtonian relaxation and Rayleigh friction, following Held and Suarez (1994)
"""
struct HeldSuarezForcingTutorial{PV <: Union{Momentum, Energy}} <:
       TendencyDef{Source, PV} end

HeldSuarezForcingTutorial() =
    (HeldSuarezForcingTutorial{Momentum}(), HeldSuarezForcingTutorial{Energy}())

filter_source(pv::PV, m, s::HeldSuarezForcingTutorial{PV}) where {PV} = s
atmos_source!(::HeldSuarezForcingTutorial, args...) = nothing

function held_suarez_forcing_coefficients(bl, args)
    @unpack state, aux = args
    @unpack ts = args.precomputed
    FT = eltype(state)

    # Parameters
    T_ref = FT(255)

    _R_d = FT(R_d(bl.param_set))
    _day = FT(day(bl.param_set))
    _grav = FT(grav(bl.param_set))
    _cp_d = FT(cp_d(bl.param_set))
    _p0 = FT(MSLP(bl.param_set))

    # Held-Suarez parameters
    k_a = FT(1 / (40 * _day))
    k_f = FT(1 / _day)
    k_s = FT(1 / (4 * _day))
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)
    σ_b = FT(7 / 10)

    # Held-Suarez forcing
    φ = latitude(bl, aux)
    p = air_pressure(ts)

    ##TODO: replace _p0 with dynamic surface pressure in Δσ calculations to account
    ##for topography, but leave unchanged for calculations of σ involved in T_equil
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor
    return (k_v = k_v, k_T = k_T, T_equil = T_equil)
end

function source(s::HeldSuarezForcingTutorial{Energy}, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    nt = held_suarez_forcing_coefficients(m, args)
    _cv_d = FT(cv_d(m.param_set))
    @unpack k_T, T_equil = nt
    T = air_temperature(ts)
    return -k_T * state.ρ * _cv_d * (T - T_equil)
end

function source(s::HeldSuarezForcingTutorial{Momentum}, m, args)
    @unpack state, aux = args
    nt = held_suarez_forcing_coefficients(m, args)
    return -nt.k_v * projection_tangential(m, aux, state.ρu)
end

function init_heldsuarez!(problem, balance_law, state, aux, localgeo, time)
    FT = eltype(state)

    # Set initial state to reference state with random perturbation
    rnd = FT(1 + rand(Uniform(-1e-3, 1e-3)))
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = rnd * aux.ref_state.ρe
end;

ClimateMachine.init();

const FT = Float32;

temp_profile_ref = DecayingTemperatureProfile{FT}(param_set)
ref_state = HydrostaticState(temp_profile_ref);

domain_height = FT(30e3)               # height of the computational domain (m)
z_sponge = FT(12e3)                    # height at which sponge begins (m)
α_relax = FT(1 / 60 / 15)              # sponge relaxation rate (1/s)
exponent = FT(2)                       # sponge exponent for squared-sinusoid profile
u_relax = SVector(FT(0), FT(0), FT(0)) # relaxation velocity (m/s)
sponge = RayleighSponge(FT, domain_height, z_sponge, α_relax, u_relax, exponent);

c_smag = FT(0.21);   # Smagorinsky constant
τ_hyper = FT(4 * 3600); # hyperdiffusion time scale
turbulence_model = SmagorinskyLilly(c_smag);
hyperdiffusion_model = DryBiharmonic(FT(4 * 3600));

model = AtmosModel{FT}(
    AtmosGCMConfigType,
    param_set;
    init_state_prognostic = init_heldsuarez!,
    ref_state = ref_state,
    turbulence = turbulence_model,
    hyperdiffusion = hyperdiffusion_model,
    moisture = DryModel(),
    source = (Gravity(), Coriolis(), HeldSuarezForcingTutorial()..., sponge),
);

poly_order = 5;                        ## discontinuous Galerkin polynomial order
n_horz = 2;                            ## horizontal element number
n_vert = 2;                            ## vertical element number
resolution = (n_horz, n_vert)
n_days = 0.1;                          ## experiment day number
timestart = FT(0);                     ## start time (s)
timeend = FT(n_days * day(param_set)); ## end time (s);

driver_config = ClimateMachine.AtmosGCMConfiguration(
    "HeldSuarez",
    poly_order,
    resolution,
    domain_height,
    param_set,
    init_heldsuarez!;
    model = model,
);

ode_solver_type = ClimateMachine.IMEXSolverType(
    splitting_type = HEVISplitting(),
    implicit_model = AtmosAcousticGravityLinearModel,
    implicit_solver = ManyColumnLU,
    solver_method = ARK2GiraldoKellyConstantinescu,
);

solver_config = ClimateMachine.SolverConfiguration(
    timestart,
    timeend,
    driver_config,
    Courant_number = FT(0.1),
    ode_solver_type = ode_solver_type,
    init_on_cpu = true,
    CFL_direction = HorizontalDirection(),
    diffdir = HorizontalDirection(),
);

filterorder = 10;
filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder);
cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
    Filters.apply!(
        solver_config.Q,
        AtmosFilterPerturbations(model),
        solver_config.dg.grid,
        filter,
        state_auxiliary = solver_config.dg.state_auxiliary,
    )
    nothing
end;

interval = "1000steps";
_planet_radius = FT(planet_radius(param_set));
info = driver_config.config_info;
boundaries = [
    FT(-90.0) FT(-180.0) _planet_radius
    FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
];
resolution = (FT(10), FT(10), FT(1000)); # in (deg, deg, m)
interpol = ClimateMachine.InterpolationConfiguration(
    driver_config,
    boundaries,
    resolution,
);

dgngrps = [
    setup_dump_state_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    ),
    setup_dump_aux_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    ),
];
dgn_config = ClimateMachine.DiagnosticsConfiguration(dgngrps);

result = ClimateMachine.invoke!(
    solver_config;
    diagnostics_config = dgn_config,
    user_callbacks = (cbfilter,),
    check_euclidean_distance = true,
);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

