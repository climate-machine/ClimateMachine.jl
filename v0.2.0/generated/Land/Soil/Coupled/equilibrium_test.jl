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
FT = Float64;

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "tutorials",
    "Land",
    "Soil",
    "interpolation_helper.jl",
));

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
bottom_water_state = nothing;

surface_heat_flux = (aux, t) -> eltype(aux)(0.0)
bottom_heat_flux = (aux, t) -> eltype(aux)(0.0)
surface_heat_state = nothing
bottom_heat_state = nothing;

function init_soil!(land, state, aux, coordinates, time)
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
    dirichlet_bc = Dirichlet(
        surface_state = surface_water_state,
        bottom_state = bottom_water_state,
    ),
    neumann_bc = Neumann(
        surface_flux = surface_water_flux,
        bottom_flux = bottom_water_flux,
    ),
);

soil_heat_model = SoilHeatModel(
    FT;
    initialT = T_init,
    dirichlet_bc = Dirichlet(
        surface_state = surface_heat_state,
        bottom_state = bottom_heat_state,
    ),
    neumann_bc = Neumann(
        surface_flux = surface_heat_flux,
        bottom_flux = bottom_heat_flux,
    ),
);

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

all_data = Dict([k => Dict() for k in 1:n_outputs]...)

K∇h_vert_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :water)[3]
κ∇T_vert_ind = varsindex(vars_state(m, GradientFlux(), FT), :soil, :heat)[3]
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
T_ind = varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)

t = ODESolvers.gettime(solver_config.solver)
thegrid = solver_config.dg.grid
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;
grads = solver_config.dg.state_gradient_flux
ϑ_l = Q[:, ϑ_l_ind, :][:]
z = aux[:, z_ind, :][:]
T = aux[:, T_ind, :][:];

initial_state = Dict{String, Array}(
    "t" => [t],
    "ϑ_l" => ϑ_l,
    "T" => T,
    "K∇h_vert" => [nothing],
    "κ∇T_vert" => [nothing],
);

zres = FT(0.02)
boundaries = [
    FT(0) FT(0) zmin
    FT(1) FT(1) zmax
]
resolution = (FT(2), FT(2), zres)
thegrid = solver_config.dg.grid
intrp_brck = create_interpolation_grid(boundaries, resolution, thegrid)
step = [1];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
    iQ, iaux, igrads = interpolate_variables((Q, aux, grads), intrp_brck)
    ϑ_l = iQ[:, ϑ_l_ind, :][:]
    T = iaux[:, T_ind, :][:]
    K∇h_vert = igrads[:, K∇h_vert_ind, :][:]
    κ∇T_vert = igrads[:, κ∇T_vert_ind, :][:]
    all_vars = Dict{String, Array}(
        "t" => [t],
        "ϑ_l" => ϑ_l,
        "T" => T,
        "K∇h_vert" => K∇h_vert,
        "κ∇T_vert" => κ∇T_vert,
    )
    all_data[step[1]] = all_vars

    step[1] += 1
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

t = ODESolvers.gettime(solver_config.solver)
iQ, iaux, igrads = interpolate_variables((Q, aux, grads), intrp_brck)

ϑ_l = iQ[:, ϑ_l_ind, :][:]
T = iaux[:, T_ind, :][:]
K∇h_vert = igrads[:, K∇h_vert_ind, :][:]
κ∇T_vert = igrads[:, κ∇T_vert_ind, :][:]
all_vars = Dict{String, Array}(
    "t" => [t],
    "ϑ_l" => ϑ_l,
    "T" => T,
    "K∇h_vert" => K∇h_vert,
    "κ∇T_vert" => κ∇T_vert,
)
all_data[n_outputs] = all_vars
iz = iaux[:, z_ind, :][:]

t = [all_data[k]["t"][1] for k in 1:n_outputs]
t = ceil.(Int64, t ./ 60)

ϑ_plot =
    plot(initial_state["ϑ_l"], z, label = "t = 0", ylabel = "z", xlabel = "ϑ_l")
plot!(all_data[1]["ϑ_l"], iz, label = "t = 0.75 days")
plot!(all_data[2]["ϑ_l"], iz, label = "t = 1.5 days")
plot!(all_data[3]["ϑ_l"], iz, label = "t = 2.25 days")
plot!(all_data[4]["ϑ_l"], iz, label = "t = 3 days")


K∇h_z_plot = plot(
    all_data[1]["K∇h_vert"],
    iz,
    label = "0.75 days",
    xlabel = "K∇h_z (m/s)",
)
plot!(all_data[2]["K∇h_vert"], iz, label = "1.5 days")
plot!(all_data[3]["K∇h_vert"], iz, label = "2.25 days")
plot!(all_data[4]["K∇h_vert"], iz, label = "3 days")
plot!(legend = :bottomleft)
plot(ϑ_plot, K∇h_z_plot)
savefig("eq_moisture_plot.png")

T_plot =
    plot(initial_state["T"], z, label = "t = 0", ylabel = "z", xlabel = "T (K)")
plot!(all_data[1]["T"], iz, label = "t = 0.75 days")
plot!(all_data[2]["T"], iz, label = "t = 1.5 days")
plot!(all_data[3]["T"], iz, label = "t = 2.25 days")
plot!(all_data[4]["T"], iz, label = "t = 3 days")
plot!(legend = :bottomright)

κ∇T_z_plot = plot(
    all_data[1]["κ∇T_vert"],
    iz,
    label = "0.75 days",
    xlabel = "κ∇T_z (W/m^2)",
)
plot!(all_data[2]["κ∇T_vert"], iz, label = "1.5 days")
plot!(all_data[3]["κ∇T_vert"], iz, label = "2.25 days")
plot!(all_data[4]["κ∇T_vert"], iz, label = "3 days")
plot!(legend = :bottomright)
plot(T_plot, κ∇T_z_plot)
savefig("eq_temperature_plot.png")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

