using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Plots

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Diagnostics
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();

ClimateMachine.init()
const FT = Float64;

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

porosity = FT(0.395);

ν_ss_quartz = FT(0.92)
ν_ss_minerals = FT(0.08)
ν_ss_om = FT(0.0)
ν_ss_gravel = FT(0.0);

Ksat = FT(4.42 / 3600 / 100) # m/s
S_s = FT(1e-3) #inverse meters
vg_n = FT(1.89)
vg_α = FT(7.5); # inverse meters

κ_quartz = FT(7.7) # W/m/K
κ_minerals = FT(2.5) # W/m/K
κ_om = FT(0.25) # W/m/K
κ_liq = FT(0.57) # W/m/K
κ_ice = FT(2.29); # W/m/K

ρp = FT(2700); # kg/m^3

κ_solid = k_solid(ν_ss_om, ν_ss_quartz, κ_quartz, κ_minerals, κ_om)
κ_sat_frozen = ksat_frozen(κ_solid, porosity, κ_ice)
κ_sat_unfrozen = ksat_unfrozen(κ_solid, porosity, κ_liq);

ρc_ds = FT((1 - porosity) * 1.926e06); # J/m^3/K

soil_param_functions = SoilParamFunctions{FT}(
    Ksat = Ksat,
    S_s = S_s,
    porosity = porosity,
    ν_ss_gravel = ν_ss_gravel,
    ν_ss_om = ν_ss_om,
    ν_ss_quartz = ν_ss_quartz,
    ρc_ds = ρc_ds,
    ρp = ρp,
    κ_solid = κ_solid,
    κ_sat_unfrozen = κ_sat_unfrozen,
    κ_sat_frozen = κ_sat_frozen,
);

function T_init(aux)
    FT = eltype(aux)
    zmax = FT(0)
    zmin = FT(-1)
    T_max = FT(289.0)
    T_min = FT(288.0)
    c = FT(20.0)
    z = aux.z
    output = T_min + (T_max - T_min) * exp(-(z - zmax) / (zmin - zmax) * c)
    return output
end;

function ϑ_l0(aux)
    FT = eltype(aux)
    zmax = FT(0)
    zmin = FT(-1)
    theta_max = FT(porosity * 0.5)
    theta_min = FT(porosity * 0.4)
    c = FT(20.0)
    z = aux.z
    output =
        theta_min +
        (theta_max - theta_min) * exp(-(z - zmax) / (zmin - zmax) * c)
    return output
end;

surface_water_flux = (aux, t) -> eltype(aux)(0.0)
bottom_water_flux = (aux, t) -> eltype(aux)(0.0)
surface_water_state = nothing
bottom_water_state = nothing
water_bc = GeneralBoundaryConditions(
    Dirichlet(
        surface_state = surface_water_state,
        bottom_state = bottom_water_state,
    ),
    Neumann(surface_flux = surface_water_flux, bottom_flux = bottom_water_flux),
);

surface_heat_flux = (aux, t) -> eltype(aux)(0.0)
bottom_heat_flux = (aux, t) -> eltype(aux)(0.0)
surface_heat_state = nothing
bottom_heat_state = nothing
heat_bc = GeneralBoundaryConditions(
    Dirichlet(
        surface_state = surface_heat_state,
        bottom_state = bottom_heat_state,
    ),
    Neumann(surface_flux = surface_heat_flux, bottom_flux = bottom_heat_flux),
);

function init_soil!(land, state, aux, localgeo, time)
    myFT = eltype(state)
    ϑ_l = myFT(land.soil.water.initialϑ_l(aux))
    θ_i = myFT(land.soil.water.initialθ_i(aux))
    state.soil.water.ϑ_l = ϑ_l
    state.soil.water.θ_i = θ_i

    θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
    ρc_ds = land.soil.param_functions.ρc_ds
    ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)

    state.soil.heat.ρe_int = volumetric_internal_energy(
        θ_i,
        ρc_s,
        land.soil.heat.initialT(aux),
        land.param_set,
    )
end;

soil_water_model = SoilWaterModel(
    FT;
    viscosity_factor = TemperatureDependentViscosity{FT}(),
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(α = vg_α, n = vg_n),
    initialϑ_l = ϑ_l0,
    boundaries = water_bc,
);

soil_heat_model = SoilHeatModel(FT; initialT = T_init, boundaries = heat_bc);

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

sources = ();

m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil!,
);

N_poly = 1
nelem_vert = 50
zmin = FT(-1)
zmax = FT(0)

driver_config = ClimateMachine.SingleStackConfiguration(
    "LandModel",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m;
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
)

t0 = FT(0)
timeend = FT(60 * 60 * 72)
dt = FT(30.0)


solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

const n_outputs = 4
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

state_types = (Prognostic(), Auxiliary(), GradientFlux())
dons_arr = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

dons = dict_of_nodal_states(solver_config, state_types; interp = true)
push!(dons_arr, dons)
push!(time_data, gettime(solver_config.solver));

z = get_z(solver_config.dg.grid; rm_dupes = true);

output_dir = @__DIR__;

mkpath(output_dir);

export_plot(
    z,
    time_data ./ (60 * 60 * 24),
    dons_arr,
    ("soil.water.ϑ_l",),
    joinpath(output_dir, "eq_moisture_plot.png");
    xlabel = "ϑ_l",
    ylabel = "z (cm)",
    time_units = "(days)",
)

export_plot(
    z,
    time_data[2:end] ./ (60 * 60 * 24),
    dons_arr[2:end],
    ("soil.water.K∇h[3]",),
    joinpath(output_dir, "eq_hydraulic_head_plot.png");
    xlabel = "K∇h (m/s)",
    ylabel = "z (cm)",
    time_units = "(days)",
)

export_plot(
    z,
    time_data ./ (60 * 60 * 24),
    dons_arr,
    ("soil.heat.T",),
    joinpath(output_dir, "eq_temperature_plot.png");
    xlabel = "T (K)",
    ylabel = "z (cm)",
    time_units = "(days)",
)

export_plot(
    z,
    time_data[2:end] ./ (60 * 60 * 24),
    dons_arr[2:end],
    ("soil.heat.κ∇T[3]",),
    joinpath(output_dir, "eq_heat_plot.png");
    xlabel = "κ∇T",
    ylabel = "z (cm)",
    time_units = "(days)",
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

