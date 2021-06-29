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
bottom_flux = (aux, t) -> eltype(aux)(0.0)
surface_state = nothing
bottom_state = nothing

ϑ_l0 = (aux) -> eltype(aux)(0.494);

soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 2.0),
    initialϑ_l = ϑ_l0,
    dirichlet_bc = Dirichlet(
        surface_state = surface_state,
        bottom_state = bottom_state,
    ),
    neumann_bc = Neumann(
        surface_flux = surface_flux,
        bottom_flux = bottom_flux,
    ),
);

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model);

sources = ();

function init_soil_water!(land, state, aux, localgeo, time)
    FT = eltype(state)
    state.soil.water.ϑ_l = FT(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = FT(land.soil.water.initialθ_i(aux))
end

m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
);

N_poly = 5;
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
timeend = FT(60 * 60 * 24 * 4)
dt = FT(5);

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

const n_outputs = 5;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

state_types = (Prognostic(), Auxiliary(), GradientFlux())
all_data = Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
time_data = FT[0] # store time data

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(all_data, dons)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

dons = dict_of_nodal_states(solver_config, state_types; interp = true)
push!(all_data, dons)
push!(time_data, gettime(solver_config.solver));

z = get_z(solver_config.dg.grid; rm_dupes = true);

slope = -1e-3

output_dir = @__DIR__;

t = time_data ./ (60 * 60 * 24);

plot(
    all_data[1]["soil.water.ϑ_l"],
    all_data[1]["z"],
    label = string("t = ", string(t[1]), "days"),
    xlim = [0.47, 0.501],
    ylabel = "z",
    xlabel = "ϑ_l",
    legend = :bottomleft,
    title = "Equilibrium test",
);
plot!(
    all_data[4]["soil.water.ϑ_l"],
    all_data[4]["z"],
    label = string("t = ", string(t[4]), "days"),
);
plot!(
    all_data[6]["soil.water.ϑ_l"],
    all_data[6]["z"],
    label = string("t = ", string(t[6]), "days"),
);
plot!(
    (all_data[1]["z"] .+ 10.0) .* slope .+ all_data[6]["soil.water.ϑ_l"][1],
    all_data[1]["z"],
    label = "expected",
);

plot!(
    1e-3 .+ all_data[1]["soil.water.ϑ_l"],
    all_data[1]["z"],
    label = "porosity",
);

savefig(joinpath(output_dir, "equilibrium_test_ϑ_l_vG.png"))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

