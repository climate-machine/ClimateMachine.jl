using ClimateMachine

ClimateMachine.init()

# Domain
Lx = 1e6   # m
Ly = 1e6   # m
Lz = 400.0 # m

# Numerical parameters
Np = 4           # Polynomial order
Ne = (25, 1, 1) # Number of elements in (x, y, z)
nothing # hide

f = 1e-4 # s⁻¹, Coriolis parameter
nothing # hide

using CLIMAParameters: AbstractEarthParameterSet, Planet
gravitational_acceleration = Planet.grav
struct EarthParameters <: AbstractEarthParameterSet end

g = gravitational_acceleration(EarthParameters()) # m s⁻²

U = 0.1            # geostrophic velocity (m s⁻¹)
L = Lx / 40        # Gaussian width (m)
a = f * U * L / g  # amplitude of the geostrophic surface displacement (m)
x₀ = Lx / 4        # Gaussian origin (m, recall that x ∈ [0, Lx])

Ψ(x, L) = exp(-x^2 / 2 * L^2) # a Gaussian

# Geostrophic ``y``-velocity
vᵍ(x, y, z) = -U * (x - x₀) / L * Ψ(x - x₀, L)

# Geostrophic surface displacement
ηᵍ(x, y, z) = a * Ψ(x - x₀, L)

ηⁱ(x, y, z) = 2 * ηᵍ(x, y, z)

using ClimateMachine.Ocean.OceanProblems: InitialConditions

initial_conditions = InitialConditions(v = vᵍ, η = ηⁱ)

@info """ Parameters for the Geostrophic adjustment problem are...

    Coriolis parameter:                            $f s⁻¹
    Gravitational acceleration:                    $g m s⁻²
    Geostrophic velocity:                          $U m s⁻¹
    Width of the initial geostrophic perturbation: $L m
    Amplitude of the initial surface perturbation: $a m
    Rossby number (U / f L):                       $(U / (f * L))

"""

using ClimateMachine.Ocean:
    Impenetrable, Penetrable, FreeSlip, Insulating, OceanBC

solid_surface_boundary_conditions = OceanBC(
    Impenetrable(FreeSlip()), # Velocity boundary conditions
    Insulating(),             # Temperature boundary conditions
)

free_surface_boundary_conditions = OceanBC(
    Penetrable(FreeSlip()),   # Velocity boundary conditions
    Insulating(),             # Temperature boundary conditions
)

boundary_conditions =
    (solid_surface_boundary_conditions, free_surface_boundary_conditions)

state_boundary_conditions = (
    (1, 1), # (west, east) boundary conditions
    (0, 0), # (south, north) boundary conditions
    (1, 2), # (bottom, top) boundary conditions
)

using ClimateMachine.Ocean.OceanProblems: InitialValueProblem

problem = InitialValueProblem{Float64}(
    dimensions = (Lx, Ly, Lz),
    initial_conditions = initial_conditions,
    boundary_conditions = boundary_conditions,
)

using ClimateMachine.Ocean.HydrostaticBoussinesq: HydrostaticBoussinesqModel

equations = HydrostaticBoussinesqModel{Float64}(
    EarthParameters(),
    problem,
    νʰ = 0.0,          # Horizontal viscosity (m² s⁻¹)
    κʰ = 0.0,          # Horizontal diffusivity (m² s⁻¹)
    fₒ = f,             # Coriolis parameter (s⁻¹)
)

driver_configuration = ClimateMachine.OceanBoxGCMConfiguration(
    "Geostrophic adjustment tutorial",    # The name of the experiment
    Np,                                   # The polynomial order
    Ne,                                   # The number of elements
    EarthParameters(),                    # The CLIMAParameters.AbstractParameterSet to use
    equations;                            # The equations to solve, represented by a `BalanceLaw`
    periodicity = (false, true, false),   # Topology of the domain
    boundary = state_boundary_conditions,
)
nothing # hide

using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids: polynomialorder

grid = driver_configuration.grid

filters = (
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1),
    exp_filter = ExponentialFilter(grid, 1, 8),
)

using ClimateMachine.ODESolvers

minute = 60.0
hours = 60minute

solver_configuration = ClimateMachine.SolverConfiguration(
    0.0,                    # start time (s)
    4hours,                 # stop time (s)
    driver_configuration,
    init_on_cpu = true,
    ode_dt = 1minute,       # time step size (s)
    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    ),
    modeldata = filters,
)

using ClimateMachine.GenericCallbacks: EveryXSimulationSteps

print_every = 10 # iterations
solver = solver_configuration.solver

tiny_progress_printer = EveryXSimulationSteps(print_every) do
    @info "Steps: $(solver.steps), time: $(solver.t), time step: $(solver.dt)"
end

using Printf
using Plots

using ClimateMachine.Ocean.Fields: assemble
using ClimateMachine.Ocean.CartesianDomains: CartesianDomain, CartesianField

# CartesianDomain and CartesianField objects to help with plotting
domain = CartesianDomain(solver_configuration.dg.grid, Ne)

u = CartesianField(domain, solver_configuration.Q, 1)
v = CartesianField(domain, solver_configuration.Q, 2)
η = CartesianField(domain, solver_configuration.Q, 3)

# Container to hold the plotted frames
movie_plots = []

plot_every = 10 # iterations

plot_maker = EveryXSimulationSteps(plot_every) do
    u_assembly = assemble(u)
    v_assembly = assemble(v)
    η_assembly = assemble(η)

    umax = 0.5 * max(maximum(abs, u), maximum(abs, v))
    ulim = (-umax, umax)

    u_plot = plot(
        u_assembly.x,
        [u_assembly.data[:, 1, 1] v_assembly.data[:, 1, 1]],
        xlim = domain.x,
        ylim = (-0.7U, 0.7U),
        label = ["u" "v"],
        linewidth = 2,
        xlabel = "x (m)",
        ylabel = "Velocities (m s⁻¹)",
    )

    η_plot = plot(
        η_assembly.x,
        η_assembly.data[:, 1, 1],
        xlim = domain.x,
        ylim = (-0.01a, 1.2a),
        linewidth = 2,
        label = nothing,
        xlabel = "x (m)",
        ylabel = "η (m)",
    )

    push!(
        movie_plots,
        (u = u_plot, η = η_plot, time = solver_configuration.solver.t),
    )

    return nothing
end

result = ClimateMachine.invoke!(
    solver_configuration;
    user_callbacks = [tiny_progress_printer, plot_maker],
)

animation = @animate for p in movie_plots
    title = @sprintf("Geostrophic adjustment at t = %.2f hours", p.time / hours)
    frame = plot(
        p.u,
        p.η,
        layout = (2, 1),
        size = (800, 600),
        title = [title ""],
    )
end

gif(animation, "geostrophic_adjustment.gif", fps = 8) # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

