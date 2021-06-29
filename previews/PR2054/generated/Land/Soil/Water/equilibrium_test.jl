using MPI
using OrderedCollections
using StaticArrays
using Statistics

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
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

const FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

soil_heat_model = PrescribedTemperatureModel();

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.495,
    Ksat = 0.0443 / (3600 * 100),
    S_s = 1e-3,
);

surface_flux = (aux, t) -> eltype(aux)(0.0)
bottom_flux = (aux, t) -> eltype(aux)(0.0);

bc = LandDomainBC(
    bottom_bc = LandComponentBC(soil_water = Neumann(bottom_flux)),
    surface_bc = LandComponentBC(soil_water = Neumann(surface_flux)),
);

ϑ_l0 = (aux) -> eltype(aux)(0.494);

soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0),
    initialϑ_l = ϑ_l0,
);

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

sources = ();

function init_soil_water!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
end

m = LandModel(
    param_set,
    m_soil;
    boundary_conditions = bc,
    source = sources,
    init_state_prognostic = init_soil_water!,
);

N_poly = 2;
nelem_vert = 20;

zmax = FT(0);
zmin = FT(-10);

driver_config = ClimateMachine.SingleStackConfiguration(
    "LandModel",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m;
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

t0 = FT(0)
timeend = FT(60 * 60 * 24 * 36)
dt = FT(100);

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

const n_outputs = 6;

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

z = get_z(solver_config.dg.grid; rm_dupes = true);

output_dir = @__DIR__;

t = time_data ./ (60 * 60 * 24);

plot(
    dons_arr[1]["soil.water.ϑ_l"],
    dons_arr[1]["z"],
    label = string("t = ", string(t[1]), "days"),
    xlim = [0.47, 0.501],
    ylabel = "z",
    xlabel = "ϑ_l",
    legend = :bottomleft,
    title = "Equilibrium test",
);
plot!(
    dons_arr[2]["soil.water.ϑ_l"],
    dons_arr[2]["z"],
    label = string("t = ", string(t[2]), "days"),
);
plot!(
    dons_arr[7]["soil.water.ϑ_l"],
    dons_arr[7]["z"],
    label = string("t = ", string(t[7]), "days"),
);
function expected(z, z_interface)
    ν = 0.495
    S_s = 1e-3
    α = 2.6
    n = 2.0
    m = 0.5
    if z < z_interface
        return -S_s * (z - z_interface) + ν
    else
        return ν * (1 + (α * (z - z_interface))^n)^(-m)
    end
end
plot!(expected.(dons_arr[1]["z"], -0.56), dons_arr[1]["z"], label = "expected");

plot!(
    1e-3 .+ dons_arr[1]["soil.water.ϑ_l"],
    dons_arr[1]["z"],
    label = "porosity",
);

savefig(joinpath(output_dir, "equilibrium_test_ϑ_l_vG.png"))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

