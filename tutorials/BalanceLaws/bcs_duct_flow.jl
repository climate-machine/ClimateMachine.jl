# # Boundary conditions tutorial (Duct flow)

# !!! note
#     This is an intermediate-level tutorial, and we
#     recommended to first review the [heat tutorial](https://clima.github.io/ClimateMachine.jl/latest/generated/Land/Heat/heat_equation/).

# In this tutorial, we'll be solving the [advection-diffusion
# equation](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation)
# in a duct as a template to experiment with boundary conditions.
# The most complex form of the advection-diffusion equations
# that we'll consider are:

# !!! danger
#     This section is not complete

# ```math
# \begin{align}
# \frac{∂ ρ}{∂ t} + ∇ ⋅ (ρu⃗) = 0 \\
# \frac{∂ ρu⃗}{∂ t} + ∇ ⋅ (ρu⃗ ⊗ u⃗ + p I) + ∇ ⋅ (-μ ∇ρu⃗) = 0 \\
# \frac{∂ ρcT}{∂ t} + ∇ ⋅ (-α ∇ρcT) = 0 \\
# \end{align}
# ```

# where
#  - `t` is time
#  - `ρ` is the density
#  - `ρu⃗` is the momentum
#  - `u⃗ = ρu⃗ / ρ` is the velocity
#  - `α` is the thermal diffusivity
#  - `T` is the temperature
#  - `c` is the heat capacity
#  - `ρcT` is the thermal energy

# We'll be experimenting with boundary conditions under several
# simplifications of the problem. Let's start with the simplest
# case: uniform density isothermal flow. In this case, the equations,
# after non-dimensionalizing, simplify to:

# ```math
# \begin{align}
# \frac{∂ u⃗}{∂ t} + ∇ ⋅ (u⃗ ⊗ u⃗) & = - ∇p + Re⁻¹ ∇ ⋅ (∇u⃗) \\
# ∇ ⋅ u⃗ & = 0 \\
# \end{align}
# ```

# To satisfy `∇ ⋅ u⃗ = 0`, we can take the divergence of the NSE, and
# derive a pressure Poisson equation, where we assume `∇ ⋅ u⃗ = 0` in
# various places:

# ```math
# \begin{align}
# \underbrace{∇ ⋅ \frac{∂ u⃗}{∂ t}}_{=0} + ∇ ⋅ (∇ ⋅ (u⃗ ⊗ u⃗)) & = ∇ ⋅ (- ∇p) + \underbrace{∇ ⋅ (Re⁻¹ ∇ ⋅ (∇u⃗))}_{=0} \\
# ∇² p & = - ∇ ⋅ (∇ ⋅ (u⃗ ⊗ u⃗)) \\
#  & = - ∂ᵢ ∂ⱼ (uᵢuⱼ) \\
#  & = - ∂ᵢ (\underbrace{uᵢ∂ⱼ uⱼ}_{=0} + uⱼ∂ⱼ uᵢ) \\
#  & = - ∂ᵢ (uⱼ∂ⱼuᵢ) \\
# \end{align}
# ```

# ClimateMachine time-steps all equations simultaneously, so
# we'll add the unsteady pressure term (`α_p⁻¹ ∂_t p`, where `α_p`
# controlls the pressure time-scale) to the equation and solve this
# by pseudo time-stepping pressure.

# Our final equations are therefore:

# ```math
# \begin{align}
# \frac{∂ u⃗}{∂ t} + ∇ ⋅ (u⃗ ⊗ u⃗) & = - ∇p + Re⁻¹ ∇²u⃗ \\
# \frac{∂ p}{∂ t} & = α_p ∇²p + α_p ∂ᵢ(uⱼ∂ⱼuᵢ) \\
# (∇ ⋅ u⃗ & = 0) \\
# \end{align}
# ```

# We can also write this in flux form (which is helpful coding things up):

# ```math
# \begin{align}
# \frac{∂ u⃗}{∂ t} + ∇ ⋅ (u⃗ ⊗ u⃗ + I p) + Re⁻¹ ∇ ⋅ (- ∇u⃗) & = 0 \\
# \frac{∂ p}{∂ t} & = ∇ ⋅ (∇p + u⃗ ⋅ ∇ u⃗) \\
# (∇ ⋅ u⃗ = 0) \\
# \end{align}
# ```

# We'll be solving these Navier-Stokes Equations (NSE) in a rectangular
# domain Ω ∈ (0 ≤ y ≤ L, -H ≤ z ≤ H):

# ![domain](https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Development_of_fluid_flow_in_the_entrance_region_of_a_pipe.jpg/1280px-Development_of_fluid_flow_in_the_entrance_region_of_a_pipe.jpg)

# with boundary conditions:

# ```math
# \begin{align}
# u⃗|_{y=0} = 1 \\
# \frac{∂u⃗}{∂y}\Big|_{y=L} = 0 \\
# u⃗|_{z=±H} = 0 \\
# p|_{y=0} = 0 \\
# \frac{∂p}{∂y}\Big|_{y=0} = 0 \\
# \frac{∂p}{∂z}\Big|_{z±H} = 0 \\
# \end{align}
# ```

# ## Code loading

using MPI
using UnPack
using OrderedCollections
using Plots
using Printf
using LinearAlgebra
using StaticArrays
using OrdinaryDiffEq
using DiffEqBase
using CLIMAParameters
# Make running locally easier from ClimateMachine.jl/:
if !("." in LOAD_PATH)
    push!(LOAD_PATH, ".")
    nothing
end

using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.CartesianDomains
using ClimateMachine.CartesianFields
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
import ClimateMachine.DGMethods.NumericalFluxes:
    normal_boundary_flux_second_order!
using ClimateMachine.BalanceLaws
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
import ClimateMachine.BalanceLaws:
    vars_state,
    prognostic_vars,
    projection,
    get_prog_state,
    eq_tends,
    source,
    wavespeed,
    flux,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_update_auxiliary_state!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    BoundaryCondition,
    boundary_conditions,
    boundary_state!

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
nothing


# ## Initialization

const FT = Float64;
ClimateMachine.init(; disable_gpu = true);
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# ## Define Partial Differential Equations (PDEs), and boilerplate kernels

# Here, we define the PDEs:

struct DuctModel{FT, APS} <: BalanceLaw
    "Parameters"
    param_set::APS
    "Inverse Reynolds number"
    Re⁻¹::FT
    "Pressure time-scale"
    α_p::FT
    "Inlet velocity magnitude"
    u_inlet::FT
    "Duct height"
    H::FT
    "Duct length"
    L::FT
    "Pressure at outlet"
    p_outlet::FT
    function DuctModel(::Type{FT};
            param_set = nothing,
            Re⁻¹::FT = FT(1 / 200),
            α_p::FT = FT(1),
            u_inlet::FT = FT(0.1),
            H::FT = FT(1),
            L::FT = FT(10),
        ) where {FT}
        @assert param_set ≠ nothing
        APS = typeof(param_set)
        p_outlet = -2*u_inlet*Re⁻¹
        return new{FT, APS}(param_set, Re⁻¹, α_p, u_inlet, H, L, p_outlet)
    end
end

model = DuctModel(FT;
    param_set = param_set,
    H = FT(1),
    L = FT(10)
);

vars_state(::DuctModel, ::Auxiliary, FT) = @vars(x::FT, y::FT, z::FT);
vars_state(::DuctModel, ::Prognostic, FT) = @vars(u⃗::SVector{3, FT}, p::FT);
vars_state(::DuctModel, ::Gradient, FT) = @vars(u⃗::SVector{3, FT}, p::FT);
vars_state(::DuctModel, ::GradientFlux, FT) =
    @vars(∇u⃗::SMatrix{3, 3, FT, 9}, ∇p::SVector{3, FT});

struct Velocity <: AbstractMomentumVariable end
struct Pressure <: AbstractPrognosticVariable end
prognostic_vars(::DuctModel) = (Velocity(), Pressure())

get_prog_state(state, ::Velocity) = (state, :u⃗)
get_prog_state(state, ::Pressure) = (state, :p)

struct Advection <: TendencyDef{Flux{FirstOrder}} end
struct PPEFlux <: TendencyDef{Flux{SecondOrder}} end
struct ∇²P <: TendencyDef{Flux{SecondOrder}} end
struct Diffusion <: TendencyDef{Flux{SecondOrder}} end
struct Pressure∇ <: TendencyDef{Flux{FirstOrder}} end
prognostic_vars(::Pressure∇) = (Velocity(),)
prognostic_vars(::Diffusion) = (Velocity(),)
prognostic_vars(::Advection) = (Velocity(), Pressure())

flux(::Velocity, ::Advection, dm::DuctModel, args) =
    args.state.u⃗ .* args.state.u⃗'

flux(::Velocity, ::Diffusion, dm::DuctModel, args) =
    -dm.Re⁻¹ * (args.diffusive.∇u⃗ .+ args.diffusive.∇u⃗')

flux(::Velocity, ::Pressure∇, dm::DuctModel, args) =
    args.state.p * I + SArray{Tuple{3, 3}}(ntuple(i -> 0, 9))

flux(::Pressure, ::∇²P, dm::DuctModel, args) = -args.diffusive.∇p .* dm.α_p

# TODO: double check
uⱼ∂ⱼuᵢ(args, i) =
    args.state.u⃗[1] * args.diffusive.∇u⃗[i, 1] +
    args.state.u⃗[2] * args.diffusive.∇u⃗[i, 2] +
    args.state.u⃗[3] * args.diffusive.∇u⃗[i, 3]

# uⱼ∂ⱼuᵢ(args, i) =
#     args.state.u⃗[1] * args.diffusive.∇u⃗[1, i] +
#     args.state.u⃗[2] * args.diffusive.∇u⃗[2, i] +
#     args.state.u⃗[3] * args.diffusive.∇u⃗[3, i]

flux(::Pressure, ::PPEFlux, dm::DuctModel, args) =
    -SVector(uⱼ∂ⱼuᵢ(args, 1), uⱼ∂ⱼuᵢ(args, 2), uⱼ∂ⱼuᵢ(args, 3)) .* dm.α_p
# flux(::Pressure, ::PPEFlux, dm::DuctModel, args) =
#     SVector(0, 0, 0) .* dm.α_p

eq_tends(pv, model, tend_def) = ()
eq_tends(::Velocity, ::DuctModel, ::Flux{FirstOrder}) =
    (Advection(), Pressure∇())
eq_tends(::Velocity, ::DuctModel, ::Flux{SecondOrder}) = (Diffusion(),)
eq_tends(::Pressure, ::DuctModel, ::Flux{SecondOrder}) = (∇²P(), PPEFlux())
eq_tends(::Pressure, ::DuctModel, ::Source) = ()

u_inlet_bc(model, aux) = (model.H - aux.z^2)*model.u_inlet

function init_state_prognostic!(
    model::DuctModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
)
    aux.x = localgeo.coord[1]
    aux.y = localgeo.coord[2]
    aux.z = localgeo.coord[3]
    FT = eltype(state)

    # Smooth everywhere
    u⃗_bc = SVector(FT(0), u_inlet_bc(model, aux)*exp(-aux.y), FT(0))
    state.u⃗ = u⃗_bc

    # if aux.y ≈ 0
    #     u⃗_bc = SVector(FT(0), u_inlet_bc(model, aux), FT(0))
    #     state.u⃗ = u⃗_bc
    # else
    #     # u⃗_bc = SVector(FT(0), u_inlet_bc(model, aux), FT(0))
    #     # state.u⃗ = u⃗_bc
    #     state.u⃗ = SVector(0, 0, 0)
    # end
    state.p = 0
    # state.p = -2*model.u_inlet*model.Re⁻¹*aux.y
    # state.p = 2*model.u_inlet*model.Re⁻¹*1000*(1 - aux.y/model.L) + model.p_outlet
    # state.p = model.p_outlet*aux.y/model.L
end;

function compute_gradient_argument!(
    model::DuctModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.u⃗ = state.u⃗
    transform.p = state.p
end;

function compute_gradient_flux!(
    model::DuctModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.∇u⃗ = ∇transform.u⃗
    diffusive.∇p = ∇transform.p
end;

# To ensure that our solution remains two-dimensional, we'll
# use the [`projection`](@ref) hook to zero-out `x`-component
# source and flux contributions:
function projection(
    pv::Velocity,
    ::DuctModel,
    ::TendencyDef{Flux{O}},
    args,
    x,
) where {O}
    return x .* SArray{Tuple{3, 3}}(0, 0, 0, 1, 1, 1, 1, 1, 1) # verified (results in umax = 0)
end
projection(::Velocity, ::DuctModel, ::TendencyDef{Source}, args, x) =
    x .* SVector(0, 1, 1)

# ### Boundary conditions

# First, we define a few boundary condition types:

struct InletBCs <: BoundaryCondition end;
struct OutletBCs <: BoundaryCondition end;
struct WallBCs <: BoundaryCondition end;

# Next, each element returned by `boundary_conditions` corresponds
# to the bc tag in `boundary = ((0, 0), (1, 2), (3, 3))` above.
# Concretely:
#  - `InletBCs` corresponds to `boundary[2][1]`
#  - `OutletBCs` corresponds to `boundary[2][2]`
#  - `WallBCs` corresponds to `boundary[3][1]` and `boundary[3][2]`
boundary_conditions(::DuctModel) = (InletBCs(), OutletBCs(), WallBCs())

# Finally, we define boundary condition kernels:

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc::InletBCs,
    model::DuctModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    u⃗_bc = SVector(0, u_inlet_bc(model, aux⁻), 0)
    state⁺.u⃗ = -state⁻.u⃗ .+ 2 * u⃗_bc
end;

function boundary_state!(
    nf::NumericalFluxGradient,
    bc::InletBCs,
    model::DuctModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    u⃗_bc = SVector(0, u_inlet_bc(model, aux⁻), 0)
    state⁺.u⃗ = u⃗_bc
end;

function normal_boundary_flux_second_order!(
    nf::NumericalFluxSecondOrder,
    bc::InletBCs,
    model::DuctModel,
    fluxᵀn::Vars{S},
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiff⁺,
    aux⁺,
    t,
    state_int⁻,
    diffusive_int⁻,
    aux_int⁻,
) where {S}
    fluxᵀn.p = 0
end
function normal_boundary_flux_second_order!(
    nf::NumericalFluxSecondOrder,
    bc::WallBCs,
    model::DuctModel,
    fluxᵀn::Vars{S},
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiff⁺,
    aux⁺,
    t,
    state_int⁻,
    diffusive_int⁻,
    aux_int⁻,
) where {S}
    fluxᵀn.p = 0
end

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc::WallBCs,
    model::DuctModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    u⃗_bc = SVector(0, 0, 0)
    state⁺.u⃗ = -state⁻.u⃗ .+ 2 * u⃗_bc # set u⃗ = 0 on walls
end;

function boundary_state!(
    nf::NumericalFluxGradient,
    bc::WallBCs,
    model::DuctModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    u⃗_bc = SVector(0, 0, 0)
    state⁺.u⃗ = u⃗_bc #
end;

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc::OutletBCs,
    model::DuctModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    # p_bc = model.p_outlet
    # state⁺.p = -state⁻.p .+ 2 * p_bc # set p = 0 at outlet

    # u⃗_bc = SVector(0, u_inlet_bc(model, aux⁻), 0)
    # state⁺.u⃗ = -state⁻.u⃗ .+ 2 * u⃗_bc
end;

function boundary_state!(
    nf::NumericalFluxGradient,
    bc::OutletBCs,
    model::DuctModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    t,
    _...,
)
    # state⁺.p = model.p_outlet

    # u⃗_bc = SVector(0, u_inlet_bc(model, aux⁻), 0)
    # state⁺.u⃗ = u⃗_bc
end;

function normal_boundary_flux_second_order!(
    nf::NumericalFluxSecondOrder,
    bc::OutletBCs,
    model::DuctModel,
    fluxᵀn::Vars{S},
    n⁻,
    state⁻,
    diffusive⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diffusive⁺,
    hyperdiff⁺,
    aux⁺,
    t,
    state_int⁻,
    diffusive_int⁻,
    aux_int⁻,
) where {S}
    # fluxᵀn.u⃗ = SVector(0, fluxᵀn.u⃗[2], 0)
    fluxᵀn.p = 0
end

# # Spatial discretization

N_poly = 5;
N_elem_x = 1; # periodic
N_elem_y = 8; # along L
N_elem_z = 3; # along H
Nx, Ny, Nz = N_elem_x * N_poly, N_elem_y * N_poly, N_elem_z * N_poly;
ymin, zmin, xmin = FT.((0, -model.H, -0.5))
ymax, zmax, xmax = FT.((model.L, +model.H, +0.5))
Δx, Δy, Δz = ((xmax - xmin) / Nx, (ymax - ymin) / Ny, (zmax - zmin) / Nz)
@show ymin, zmin, xmin
@show ymax, zmax, xmax
@show Δx, Δy, Δz

domain = RectangularDomain(
    Ne = (N_elem_x, N_elem_y, N_elem_z),
    Np = N_poly,
    x = (xmin, xmax),
    y = (ymin, ymax),
    z = (zmin, zmax),
    periodicity = (true, false, false),
)
@show domain

@inline function wavespeed(
    m::DuctModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    uN = abs(dot(nM, state.u⃗))
    ws = fill(uN, MVector{number_states(m, Prognostic()), eltype(state.u⃗)})
    return ws
end

# TODO: should we even use AtmosLESConfiguration? We need a physics-agnostic rectangular box configuration.
driver_config = ClimateMachine.AtmosLESConfiguration(
    "DuctFlow",
    N_poly,
    (Δx, Δy, Δz),
    xmax,
    ymax,
    zmax,
    param_set,
    init_state_prognostic!;
    model = model,
    xmin = xmin,
    ymin = ymin,
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    boundary = ((0, 0), (1, 2), (3, 3)),
    periodicity = (true, false, false),
    Ncutoff = N_poly - 1, # overintegration, (positively) impacts BCs...
);

# # Time discretization / solver

t0 = FT(0)

Δh_min = min(Δy, Δz) # conservative
u_max = 2 # approximate
given_Fourier = FT(1)
given_CFL = FT(1)
ν = model.Re⁻¹ # based on non-dimensionalizing

Δt_Fourier = given_Fourier * Δh_min^2 / ν
Δt_CFL = given_CFL * Δy / u_max

ode_dt = min(Δt_CFL, Δt_Fourier) / 100

## fixed_number_of_steps = 100_000
## fixed_number_of_steps = 50_000
# fixed_number_of_steps = 30_000
# fixed_number_of_steps = 25_000
fixed_number_of_steps = 100
timeend = fixed_number_of_steps * ode_dt
@show Δt_CFL
@show Δt_Fourier
@show ode_dt
@show timeend
## timeend = FT(40)

solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config;
    ode_dt = ode_dt,
    fixed_number_of_steps = fixed_number_of_steps,
    # ode_solver_type = ExplicitSolverType(),
    ode_solver_type = ExplicitSolverType(; solver_method = LSRKEulerMethod),
)

Q = solver_config.Q;
grid = solver_config.dg.grid;

# Define some diagnostic fields:

vs_prog = vars_state(model, Prognostic(), FT)
u = SpectralElementField(domain, grid, Q, varsindex(vs_prog, :u⃗)[1])
v = SpectralElementField(domain, grid, Q, varsindex(vs_prog, :u⃗)[2])
w = SpectralElementField(domain, grid, Q, varsindex(vs_prog, :u⃗)[3])
p = SpectralElementField(domain, grid, Q, varsindex(vs_prog, :p)[1])

fields = (u = u, v = v, w = w, p = p)
fetched_states = []
nothing


# Define the number of outputs from `t0` to `timeend`
n_outputs = 15
# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
every_x_simulation_time = timeend / n_outputs
@show every_x_simulation_time

data_fetcher = EveryXSimulationTime(every_x_simulation_time) do
    push!(
        fetched_states,
        (
            u = assemble(fields.u.elements),
            v = assemble(fields.v.elements),
            w = assemble(fields.w.elements),
            p = assemble(fields.p.elements),
            time = gettime(solver_config.solver),
        ),
    )
    return nothing
end

print_every = 1000 # iterations
wall_clock = [time_ns()]

tiny_progress_printer = EveryXSimulationSteps(print_every) do
    max_v = maximum(abs, fields.v)
    @info(@sprintf(
        "Steps: %d, time: %.2f, Δt: %.2f, max(|v|): %.2e, elapsed time: %.2f secs",
        ODESolvers.getsteps(solver_config.solver),
        ODESolvers.gettime(solver_config.solver),
        ODESolvers.getdt(solver_config.solver),
        max_v,
        1e-9 * (time_ns() - wall_clock[1])
    ))
    isfinite(max_v) || error("max_v is not finite")

    wall_clock[1] = time_ns()
end

# Hack to try to substepping PPE:
n_substeps = 10
i_frozen_vars = (
    varsindex(vs_prog, :u⃗)[1],
    varsindex(vs_prog, :u⃗)[2],
    varsindex(vs_prog, :u⃗)[3],
)
Q_freeze = deepcopy(solver_config.Q)
freeze_momentum(n_substeps, nstep) = mod(nstep, n_substeps) ≠ 0
cb_freeze_momentum = EveryXSimulationSteps(1) do
    nstep = getsteps(solver_config.solver)
    if freeze_momentum(n_substeps, nstep) # undo update in solve!
        for i in i_frozen_vars
            solver_config.Q[:, i, :] .= Q_freeze[:, i, :]
        end
    else # update Q_freeze
        Q_freeze .= solver_config.Q
    end
end

# push!(
#     fetched_states,
#     (
#         u = assemble(fields.u.elements),
#         v = assemble(fields.v.elements),
#         w = assemble(fields.w.elements),
#         p = assemble(fields.p.elements),
#         time = gettime(solver_config.solver),
#     ),
# )

result = ClimateMachine.invoke!(
    solver_config;
    user_callbacks = [tiny_progress_printer, data_fetcher, cb_freeze_momentum],
    # user_callbacks = [tiny_progress_printer, data_fetcher],
)

umax = maximum([maximum(state.u.data) for state in fetched_states])
vmax = maximum([maximum(state.v.data) for state in fetched_states])
wmax = maximum([maximum(state.w.data) for state in fetched_states])
pmax = maximum([maximum(state.p.data) for state in fetched_states])
umin = minimum([minimum(state.u.data) for state in fetched_states])
vmin = minimum([minimum(state.v.data) for state in fetched_states])
wmin = minimum([minimum(state.w.data) for state in fetched_states])
pmin = minimum([minimum(state.p.data) for state in fetched_states])

ulim = (umin, umax)
vlim = (vmin, vmax)
wlim = (wmin, wmax)
plim = (pmin, pmax)

@show ulim
@show vlim
@show wlim
@show plim

ulevels = range(ulim[1], ulim[2], length = 31)
vlevels = range(vlim[1], vlim[2], length = 31)
wlevels = range(wlim[1], wlim[2], length = 31)
plevels = range(plim[1], plim[2], length = 31)
@show umax

animation = @animate for (i, state) in enumerate(fetched_states)
    @info "Plotting frame $i of $(length(fetched_states)) at time $(state.time)"

    x, y, z = state.v.x[:, 1, 1], state.v.y[1, :, 1], state.v.z[1, 1, :]
    u_plane = state.u.data[1, :, :]
    v_plane = state.v.data[1, :, :]
    w_plane = state.w.data[1, :, :]
    p_plane = state.p.data[1, :, :]

    u_plot = contourf(
        y,
        z,
        u_plane';
        linewidth = 0,
        xlim = domain.y,
        ylim = domain.z,
        xlabel = "y",
        ylabel = "z",
        color = :balance,
        colorbar = true,
        clim = ulim,
        levels = ulevels,
        title = @sprintf("u at t = %.2f", state.time),
    )

    v_plot = contourf(
        y,
        z,
        v_plane';
        linewidth = 0,
        xlim = domain.y,
        ylim = domain.z,
        xlabel = "y",
        ylabel = "z",
        color = :balance,
        colorbar = true,
        clim = vlim,
        levels = vlevels,
        title = @sprintf("v at t = %.2f", state.time),
    )

    w_plot = contourf(
        y,
        z,
        w_plane';
        linewidth = 0,
        xlim = domain.y,
        ylim = domain.z,
        xlabel = "y",
        ylabel = "z",
        color = :balance,
        colorbar = true,
        clim = wlim,
        levels = wlevels,
        title = @sprintf("w at t = %.2f", state.time),
    )

    p_plot = contourf(
        y,
        z,
        p_plane';
        linewidth = 0,
        xlim = domain.y,
        ylim = domain.z,
        xlabel = "y",
        ylabel = "z",
        color = :balance,
        colorbar = true,
        clim = plim,
        levels = plevels,
        title = @sprintf("p at t = %.2f", state.time),
    )

    plot(
        v_plot,
        w_plot,
        p_plot,
        layout = Plots.grid(3, 1, heights = (0.3, 0.3, 1 - 2 * 0.3)),
        link = :x,
        size = (600, 400),
    )

    # plot(
    #     v_plot,
    #     w_plot,
    #     layout = Plots.grid(2, 1, heights = (0.5, 0.5)),
    #     link = :x,
    #     size = (600, 400),
    # )

    # plot(v_plot,
    #     layout = Plots.grid(1, 1, heights = (1.0,)),
    #     link = :x,
    #     size = (600, 400),
    # )
end

gif(animation, "duct_flow.gif", fps = 5) # hide
nothing

# ![Duct Flow](duct_flow.gif)
