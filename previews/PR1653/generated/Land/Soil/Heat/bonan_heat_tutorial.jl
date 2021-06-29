using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using Plots
using DelimitedFiles

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
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
import ClimateMachine.DGMethods: calculate_dt
using ClimateMachine.ArtifactWrappers

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();

ClimateMachine.init()
FT = Float32;

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "tutorials",
    "Land",
    "Soil",
    "interpolation_helper.jl",
));

ν_ss_silt_array =
    FT.(
        [5.0, 12.0, 32.0, 70.0, 39.0, 15.0, 56.0, 34.0, 6.0, 47.0, 20.0] ./
        100.0,
    )
ν_ss_quartz_array =
    FT.(
        [92.0, 82.0, 58.0, 17.0, 43.0, 58.0, 10.0, 32.0, 52.0, 6.0, 22.0] ./
        100.0,
    )
ν_ss_clay_array =
    FT.(
        [3.0, 6.0, 10.0, 13.0, 18.0, 27.0, 34.0, 34.0, 42.0, 47.0, 58.0] ./
        100.0,
    )
porosity_array =
    FT.([
        0.395,
        0.410,
        0.435,
        0.485,
        0.451,
        0.420,
        0.477,
        0.476,
        0.426,
        0.492,
        0.482,
    ]);

soil_type_index = 1
ν_ss_minerals =
    ν_ss_clay_array[soil_type_index] + ν_ss_silt_array[soil_type_index]
ν_ss_quartz = ν_ss_quartz_array[soil_type_index]
porosity = porosity_array[soil_type_index];

ν_ss_om = FT(0.0)
ν_ss_gravel = FT(0.0);

κ_quartz = FT(7.7) # W/m/K
κ_minerals = FT(2.5) # W/m/K
κ_om = FT(0.25) # W/m/K
κ_liq = FT(0.57) # W/m/K
κ_ice = FT(2.29); # W/m/K

ρp = FT(2700) # kg/m^3
κ_solid = k_solid(ν_ss_om, ν_ss_quartz, κ_quartz, κ_minerals, κ_om)
κ_sat_frozen = ksat_frozen(κ_solid, porosity, κ_ice)
κ_sat_unfrozen = ksat_unfrozen(κ_solid, porosity, κ_liq);

ρc_ds = FT((1 - porosity) * 1.926e06) # J/m^3/K

soil_param_functions = SoilParamFunctions{FT}(
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

prescribed_augmented_liquid_fraction = FT(porosity * 0.8)
prescribed_volumetric_ice_fraction = FT(0.0);

heat_surface_state = (aux, t) -> eltype(aux)(288.15)
heat_bottom_flux = (aux, t) -> eltype(aux)(0.0)
T_init = (aux) -> eltype(aux)(275.15);

function init_soil!(land, state, aux, localgeo, time)
    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, time)
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

soil_water_model = PrescribedWaterModel(
    (aux, t) -> prescribed_augmented_liquid_fraction,
    (aux, t) -> prescribed_volumetric_ice_fraction,
);

soil_heat_model = SoilHeatModel(
    FT;
    initialT = T_init,
    dirichlet_bc = Dirichlet(
        surface_state = heat_surface_state,
        bottom_state = nothing,
    ),
    neumann_bc = Neumann(
        surface_flux = nothing,
        bottom_flux = heat_bottom_flux,
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
nelem_vert = 100

zmax = FT(0)
zmin = FT(-1)

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

function calculate_dt(dg, model::LandModel, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = DGMethods.courant(diffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end

function diffusive_courant(
    m::LandModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    soil = m.soil
    ϑ_l, θ_i = get_water_content(soil.water, aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, soil.param_functions.porosity)
    κ_dry = k_dry(m.param_set, soil.param_functions)
    S_r = relative_saturation(θ_l, θ_i, soil.param_functions.porosity)
    kersten = kersten_number(θ_i, S_r, soil.param_functions)
    κ_sat = saturated_thermal_conductivity(
        θ_l,
        θ_i,
        soil.param_functions.κ_sat_unfrozen,
        soil.param_functions.κ_sat_frozen,
    )
    κ = thermal_conductivity(κ_dry, kersten, κ_sat)
    ρc_ds = soil.param_functions.ρc_ds
    ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, m.param_set)
    return Δt * κ / (Δx * Δx * ρc_ds)
end


t0 = FT(0)
timeend = FT(60 * 60 * 3)
Courant_number = FT(0.5) # much bigger than this leads to domain errors

solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config;
    Courant_number = Courant_number,
    CFL_direction = VerticalDirection(),
);

ClimateMachine.invoke!(solver_config);

aux = solver_config.dg.state_auxiliary;

zres = FT(0.02)
boundaries = [
    FT(0) FT(0) zmin
    FT(1) FT(1) zmax
]
resolution = (FT(2), FT(2), zres)
thegrid = solver_config.dg.grid
intrp_brck = create_interpolation_grid(boundaries, resolution, thegrid);

iaux = interpolate_variables([(aux)], intrp_brck)
iaux = iaux[1]
z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
iz = Array(iaux[:, z_ind, :][:])
z = Array(aux[:, z_ind, :][:])
T_ind = varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)
iT = Array(iaux[:, T_ind, :][:])

plot(
    iT,
    iz,
    label = "ClimateMachine",
    ylabel = "z (m)",
    xlabel = "T (K)",
    title = "Heat transfer in sand",
)
plot!(T_init.(z), z, label = "Initial condition")
filename = "bonan_heat_data.csv"
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
bonan_dataset = ArtifactWrapper(
    joinpath(clima_dir, "tutorials", "Land", "Soil", "Artifacts.toml"),
    "bonan_soil_heat",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/99vm8q8tlyoulext6c35lnd3355tx6bu.csv",
        filename = filename,
    ),],
)
bonan_dataset_path = get_data_folder(bonan_dataset)
data = joinpath(bonan_dataset_path, filename)
ds_bonan = readdlm(data, ',')
bonan_T = reverse(ds_bonan[:, 2])
bonan_z = reverse(ds_bonan[:, 1])
bonan_T_continuous = Spline1D(bonan_z, bonan_T)
bonan_at_clima_z = bonan_T_continuous.(z)
plot!(bonan_at_clima_z, z, label = "Bonan simulation")
plot!(legend = :bottomleft)
savefig("thermal_conductivity_comparison.png")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

