using MPI
using OrderedCollections
using Plots
using StaticArrays
using OrdinaryDiffEq
using DiffEqBase

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils

import ClimateMachine.BalanceLaws:
    vars_state,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_update_auxiliary_state!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    boundary_state!

import ClimateMachine.DGMethods: calculate_dt

FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "docs", "plothelpers.jl"));

Base.@kwdef struct HeatModel{FT, APS} <: BalanceLaw
    "Parameters"
    param_set::APS
    "Heat capacity"
    ρc::FT = 1
    "Thermal diffusivity"
    α::FT = 0.01
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Bottom boundary value for temperature (Dirichlet boundary conditions)"
    T_bottom::FT = 300.0
    "Top flux (α∇ρcT) at top boundary (Neumann boundary conditions)"
    flux_top::FT = 0.0
end

m = HeatModel{FT, typeof(param_set)}(; param_set = param_set);

vars_state(::HeatModel, ::Auxiliary, FT) = @vars(z::FT, T::FT);

vars_state(::HeatModel, ::Prognostic, FT) = @vars(ρcT::FT);

vars_state(::HeatModel, ::Gradient, FT) = @vars(ρcT::FT);

vars_state(::HeatModel, ::GradientFlux, FT) = @vars(α∇ρcT::SVector{3, FT});

function nodal_init_state_auxiliary!(
    m::HeatModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;

function init_state_prognostic!(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    state.ρcT = m.ρc * aux.T
end;

function nodal_update_auxiliary_state!(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    aux.T = state.ρcT / m.ρc
end;

function compute_gradient_argument!(
    m::HeatModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρcT = state.ρcT
end;

function compute_gradient_flux!(
    m::HeatModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.α∇ρcT = -m.α * ∇transform.ρcT
end;

function source!(m::HeatModel, _...) end;
function flux_first_order!(m::HeatModel, _...) end;

function flux_second_order!(
    m::HeatModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρcT += diffusive.α∇ρcT
end;

function boundary_state!(
    nf,
    m::HeatModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    # Apply Dirichlet BCs
    if bctype == 1 # At bottom
        state⁺.ρcT = m.ρc * m.T_bottom
    elseif bctype == 2 # At top
        nothing
    end
end;

function boundary_state!(
    nf,
    m::HeatModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    # Apply Neumann BCs
    if bctype == 1 # At bottom
        nothing
    elseif bctype == 2 # At top
        diff⁺.α∇ρcT = n⁻ * m.flux_top
    end
end;

N_poly = 5;

nelem_vert = 10;

zmax = FT(1);

driver_config = ClimateMachine.SingleStackConfiguration(
    "HeatEquation",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

t0 = FT(0)
timeend = FT(40)

function calculate_dt(dg, model::HeatModel, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = DGMethods.courant(diffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end

function diffusive_courant(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    return Δt * m.α / (Δx * Δx)
end

use_implicit_solver = false
if use_implicit_solver
    given_Fourier = FT(30)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        ode_solver_type = ImplicitSolverType(OrdinaryDiffEq.Kvaerno3(
            autodiff = false,
            linsolve = LinSolveGMRES(),
        )),
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
else
    given_Fourier = FT(0.7)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
end;


grid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;

output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100; # convert from meters to cm
z_key = "z";
z_label = "z [cm]";
z = get_z(grid, z_scale);

all_data = Dict[dict_of_nodal_states(solver_config, [z_key])]  # store initial condition at ``t=0``
time_data = FT[0]                                      # store time data

export_plot(
    z,
    all_data,
    ("ρcT",),
    joinpath(output_dir, "initial_condition.png");
    xlabel = "ρcT",
    ylabel = z_label,
    time_data = time_data,
);

const n_outputs = 5;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    push!(all_data, dict_of_nodal_states(solver_config, [z_key]))
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

push!(all_data, dict_of_nodal_states(solver_config, [z_key]));
push!(time_data, gettime(solver_config.solver));

@show keys(all_data[1])

export_plot(
    z,
    all_data,
    ("ρcT",),
    joinpath(output_dir, "solution_vs_time.png");
    xlabel = "ρcT",
    ylabel = z_label,
    time_data = time_data,
);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

