using MPI
using OrderedCollections
using Plots
using StaticArrays

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.DGmethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils

import ClimateMachine.DGmethods:
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    boundary_state!

FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "docs", "plothelpers.jl"));

Base.@kwdef struct HeatModel{FT} <: BalanceLaw
    "Parameters"
    param_set::AbstractParameterSet = param_set
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

m = HeatModel{FT}();

vars_state_auxiliary(::HeatModel, FT) = @vars(z::FT, T::FT);

vars_state_conservative(::HeatModel, FT) = @vars(ρcT::FT);

vars_state_gradient(::HeatModel, FT) = @vars(ρcT::FT);

vars_state_gradient_flux(::HeatModel, FT) = @vars(α∇ρcT::SVector{3, FT});

function init_state_auxiliary!(m::HeatModel, aux::Vars, geom::LocalGeometry)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;

function init_state_conservative!(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    state.ρcT = m.ρc * aux.T
end;

function update_auxiliary_state!(
    dg::DGModel,
    m::HeatModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(heat_eq_nodal_update_aux!, dg, m, Q, t, elems)
    return true # TODO: remove return true
end;

function heat_eq_nodal_update_aux!(
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
function flux_first_order!(
    m::HeatModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end;

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
    if bctype == 1 # bottom
        state⁺.ρcT = m.ρc * m.T_bottom
    elseif bctype == 2 # top
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
    if bctype == 1 # bottom
        state⁺.ρcT = m.ρc * m.T_bottom
    elseif bctype == 2 # top
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

Δ = min_node_distance(driver_config.grid)

given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 / m.α;
dt = Fourier_bound

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
grid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;

output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100 # convert from meters to cm
z_key = "z"
z_label = "z [cm]"
z = get_z(grid, z_scale)
state_vars = SingleStackUtils.get_vars_from_nodal_stack(
    grid,
    Q,
    vars_state_conservative(m, FT),
)
aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
    grid,
    aux,
    vars_state_auxiliary(m, FT),
)
all_vars = OrderedDict(state_vars..., aux_vars...);
export_plot_snapshot(
    z,
    all_vars,
    ("ρcT",),
    joinpath(output_dir, "initial_condition.png"),
    z_label,
);

const n_outputs = 5;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

all_data = Dict([k => Dict() for k in 0:n_outputs]...)
all_data[0] = all_vars # store initial condition at ``t=0``

step = [1];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    solver_config.solver,
) do (init = false)
    state_vars = SingleStackUtils.get_vars_from_nodal_stack(
        grid,
        Q,
        vars_state_conservative(m, FT),
    )
    aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
        grid,
        aux,
        vars_state_auxiliary(m, FT);
        exclude = [z_key],
    )
    all_vars = OrderedDict(state_vars..., aux_vars...)
    all_data[step[1]] = all_vars

    step[1] += 1
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

@show keys(all_data[0])

export_plot(
    z,
    all_data,
    ("ρcT",),
    joinpath(output_dir, "solution_vs_time.png"),
    z_label,
);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

