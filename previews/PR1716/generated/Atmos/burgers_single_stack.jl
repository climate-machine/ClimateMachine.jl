using MPI
using Distributions
using OrderedCollections
using Plots
using StaticArrays
using LinearAlgebra: Diagonal, tr

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
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

using ClimateMachine.Orientations:
    Orientation,
    FlatOrientation,
    init_aux!,
    vertical_unit_vector,
    projection_tangential

import ClimateMachine.BalanceLaws:
    vars_state,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    init_state_auxiliary!,
    init_state_prognostic!,
    boundary_state!

FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "docs", "plothelpers.jl"));

Base.@kwdef struct BurgersEquation{FT, APS, O} <: BalanceLaw
    "Parameters"
    param_set::APS
    "Orientation model"
    orientation::O
    "Heat capacity"
    c::FT = 1
    "Vertical dynamic viscosity"
    μv::FT = 1e-4
    "Horizontal dynamic viscosity"
    μh::FT = 1
    "Vertical thermal diffusivity"
    αv::FT = 1e-2
    "Horizontal thermal diffusivity"
    αh::FT = 1
    "IC Gaussian noise standard deviation"
    σ::FT = 5e-2
    "Rayleigh damping"
    γ::FT = 5
    "Domain height"
    zmax::FT = 1
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Bottom boundary value for temperature (Dirichlet boundary conditions)"
    T_bottom::FT = 300.0
    "Top flux (α∇ρcT) at top boundary (Neumann boundary conditions)"
    flux_top::FT = 0.0
    "Divergence damping coefficient (horizontal)"
    νd::FT = 1
end

orientation = FlatOrientation()

m = BurgersEquation{FT, typeof(param_set), typeof(orientation)}(
    param_set = param_set,
    orientation = orientation,
);

function vars_state(m::BurgersEquation, st::Auxiliary, FT)
    @vars begin
        coord::SVector{3, FT}
        orientation::vars_state(m.orientation, st, FT)
    end
end

vars_state(::BurgersEquation, ::Prognostic, FT) =
    @vars(ρ::FT, ρu::SVector{3, FT}, ρcT::FT);

vars_state(::BurgersEquation, ::Gradient, FT) =
    @vars(u::SVector{3, FT}, ρcT::FT, ρu::SVector{3, FT});

vars_state(::BurgersEquation, ::GradientFlux, FT) = @vars(
    μ∇u::SMatrix{3, 3, FT, 9},
    α∇ρcT::SVector{3, FT},
    νd∇D::SMatrix{3, 3, FT, 9}
);

function nodal_init_state_auxiliary!(
    m::BurgersEquation,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.coord = geom.coord
end;

function init_state_auxiliary!(
    m::BurgersEquation,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    init_aux!(m, m.orientation, state_auxiliary, grid, direction)

    init_state_auxiliary!(
        m,
        nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
        direction,
    )
end;

function init_state_prognostic!(
    m::BurgersEquation,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    z = aux.coord[3]
    ε1 = rand(Normal(0, m.σ))
    ε2 = rand(Normal(0, m.σ))
    state.ρ = 1
    ρu = 1 - 4 * (z - m.zmax / 2)^2 + ε1
    ρv = 1 - 4 * (z - m.zmax / 2)^2 + ε2
    ρw = 0
    state.ρu = SVector(ρu, ρv, ρw)

    state.ρcT = state.ρ * m.c * m.initialT
end;

function compute_gradient_argument!(
    m::BurgersEquation,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρcT = state.ρcT
    transform.u = state.ρu / state.ρ
    transform.ρu = state.ρu
end;

function compute_gradient_flux!(
    m::BurgersEquation{FT},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    ∇ρu = ∇transform.ρu
    ẑ = vertical_unit_vector(m.orientation, m.param_set, aux)
    divergence = tr(∇ρu) - ẑ' * ∇ρu * ẑ
    diffusive.α∇ρcT = Diagonal(SVector(m.αh, m.αh, m.αv)) * ∇transform.ρcT
    diffusive.μ∇u = Diagonal(SVector(m.μh, m.μh, m.μv)) * ∇transform.u
    diffusive.νd∇D =
        Diagonal(SVector(m.νd, m.νd, FT(0))) *
        Diagonal(SVector(divergence, divergence, FT(0)))
end;

function source!(
    m::BurgersEquation{FT},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    args...,
) where {FT}
    ẑ = vertical_unit_vector(m.orientation, m.param_set, aux)
    z = aux.coord[3]
    ρ̄ū =
        state.ρ * SVector{3, FT}(
            0.5 - 2 * (z - m.zmax / 2)^2,
            0.5 - 2 * (z - m.zmax / 2)^2,
            0.0,
        )
    ρu_p = state.ρu - ρ̄ū
    source.ρu -=
        m.γ * projection_tangential(m.orientation, m.param_set, aux, ρu_p)
end;

function flux_first_order!(
    m::BurgersEquation,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    flux.ρ = state.ρu

    u = state.ρu / state.ρ
    flux.ρu = state.ρu * u'
    flux.ρcT = u * state.ρcT
end;

function flux_second_order!(
    m::BurgersEquation,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρcT -= diffusive.α∇ρcT
    flux.ρu -= diffusive.μ∇u
    flux.ρu -= diffusive.νd∇D
end;

function boundary_state!(
    nf,
    m::BurgersEquation,
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
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        state⁺.ρcT = state⁺.ρ * m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
    end
end;

function boundary_state!(
    nf,
    m::BurgersEquation,
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
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        state⁺.ρcT = state⁺.ρ * m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        diff⁺.α∇ρcT = -n⁻ * m.flux_top
    end
end;

N_poly = 5;

nelem_vert = 10;

zmax = m.zmax;

driver_config = ClimateMachine.SingleStackConfiguration(
    "BurgersEquation",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

t0 = FT(0);
timeend = FT(1);

Δ = min_node_distance(driver_config.grid)

given_Fourier = FT(0.5);
Fourier_bound = given_Fourier * Δ^2 / max(m.αh, m.μh, m.νd);
Courant_bound = FT(0.5) * Δ;
dt = min(Fourier_bound, Courant_bound)

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100 # convert from meters to cm
z_key = "z"
z_label = "z [cm]"
z = get_z(driver_config.grid, z_scale)
state_vars = get_vars_from_nodal_stack(
    driver_config.grid,
    solver_config.Q,
    vars_state(m, Prognostic(), FT),
);

state_data = Dict[state_vars]  # store initial condition at ``t=0``
time_data = FT[0]                                      # store time data

export_plot(
    z,
    state_data,
    ("ρcT",),
    joinpath(output_dir, "initial_condition_T_nodal.png"),
    xlabel = "ρcT at southwest node",
    ylabel = z_label,
    time_data = time_data,
);
export_plot(
    z,
    state_data,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_u_nodal.png"),
    xlabel = "ρu at southwest node",
    ylabel = z_label,
    time_data = time_data,
);
export_plot(
    z,
    state_data,
    ("ρu[2]",),
    joinpath(output_dir, "initial_condition_v_nodal.png"),
    xlabel = "ρv at southwest node",
    ylabel = z_label,
    time_data = time_data,
);

state_vars_var = get_horizontal_variance(
    driver_config.grid,
    solver_config.Q,
    vars_state(m, Prognostic(), FT),
);

state_vars_avg = get_horizontal_mean(
    driver_config.grid,
    solver_config.Q,
    vars_state(m, Prognostic(), FT),
);

data_avg = Dict[state_vars_avg]
data_var = Dict[state_vars_var]

export_plot(
    z,
    data_avg,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_avg_u.png");
    xlabel = "Horizontal mean of ρu",
    ylabel = z_label,
    time_data = time_data,
);
export_plot(
    z,
    data_var,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_variance_u.png"),
    xlabel = "Horizontal variance of ρu",
    ylabel = z_label,
    time_data = time_data,
);

const n_outputs = 5;
const every_x_simulation_time = timeend / n_outputs;

dims = OrderedDict(z_key => collect(z));

data_var = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
data_var[1] = state_vars_var

data_avg = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
data_avg[1] = state_vars_avg

data_nodal = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
data_nodal[1] = state_vars

callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    state_vars_var = get_horizontal_variance(
        driver_config.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
    )
    state_vars_avg = get_horizontal_mean(
        driver_config.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
    )
    state_vars = get_vars_from_nodal_stack(
        driver_config.grid,
        solver_config.Q,
        vars_state(m, Prognostic(), FT),
        i = 1,
        j = 1,
    )
    push!(data_var, state_vars_var)
    push!(data_avg, state_vars_avg)
    push!(data_nodal, state_vars)
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

export_plot(
    z,
    data_avg,
    ("ρu[1]"),
    joinpath(output_dir, "solution_vs_time_u_avg.png"),
    xlabel = "Horizontal mean of ρu",
    ylabel = z_label,
    time_data = time_data,
);
export_plot(
    z,
    data_var,
    ("ρu[1]"),
    joinpath(output_dir, "variance_vs_time_u.png"),
    xlabel = "Horizontal variance of ρu",
    ylabel = z_label,
    time_data = time_data,
);
export_plot(
    z,
    data_avg,
    ("ρcT"),
    joinpath(output_dir, "solution_vs_time_T_avg.png"),
    xlabel = "Horizontal mean of ρcT",
    ylabel = z_label,
    time_data = time_data,
);
export_plot(
    z,
    data_var,
    ("ρcT"),
    joinpath(output_dir, "variance_vs_time_T.png"),
    xlabel = "Horizontal variance of ρcT",
    ylabel = z_label,
    time_data = time_data,
);
export_plot(
    z,
    data_nodal,
    ("ρu[1]"),
    joinpath(output_dir, "solution_vs_time_u_nodal.png"),
    xlabel = "ρu at southwest node",
    ylabel = z_label,
    time_data = time_data,
);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

