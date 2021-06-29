using MPI
using Test
using ClimateMachine
ClimateMachine.init()
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu

include(joinpath(
    @__DIR__,
    "..",
    "..",
    "..",
    "test",
    "Numerics",
    "DGmethods",
    "advection_diffusion",
    "advection_diffusion_model.jl",
))

Base.@kwdef struct SwirlingFlow{FT} <: AdvectionDiffusionProblem
    period::FT = 5
end

init_velocity_diffusion!(::SwirlingFlow, aux::Vars, geom::LocalGeometry) =
    nothing

cosbell(τ, q) = τ ≤ 1 ? ((1 + cospi(τ)) / 2)^q : zero(τ)

function initial_condition!(::SwirlingFlow, state, aux, coord, t)
    FT = eltype(state)
    x, y, _ = aux.coord
    x0, y0 = FT(1 // 4), FT(1 // 4)
    τ = 4 * hypot(x - x0, y - y0)
    state.ρ = cosbell(τ, 3)
end

has_variable_coefficients(::SwirlingFlow) = true
function update_velocity_diffusion!(
    problem::SwirlingFlow,
    ::AdvectionDiffusion,
    state::Vars,
    aux::Vars,
    t::Real,
)
    x, y, _ = aux.coord
    sx, cx = sinpi(x), cospi(x)
    sy, cy = sinpi(y), cospi(y)
    ct = cospi(t / problem.period)

    u = 2 * sx^2 * sy * cy * ct
    v = -2 * sy^2 * sx * cx * ct
    aux.u = SVector(u, v, 0)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname)
    # name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state_conservative(model, eltype(Q)))

    writevtk(filename, Q, dg, statenames)

    # generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        # name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        # name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, statenames)

        @info "Done writing VTK: $pvtuprefix"
    end
end

function run(
    mpicomm,
    ArrayType,
    topl,
    problem,
    dt,
    N,
    timeend,
    FT,
    vtkdir,
    outputtime,
)
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    model = AdvectionDiffusion{2, false, true}(problem)

    dg = DGModel(
        model,
        grid,
        UpwindNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    initialsumQ = weightedsum(Q)

    Qe = copy(Q)

    rhs! = function (dQdt, Q, ::Nothing, t; increment = false)
        Filters.apply!(Q, 1, grid, TMARFilter())
        dg(dQdt, Q, nothing, t; increment = false)
    end

    odesolver = SSPRK33ShuOsher(rhs!, Q; dt = dt, t0 = 0)

    cbTMAR = EveryXSimulationSteps(1) do
        Filters.apply!(Q, 1, grid, TMARFilter())
    end

    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    vtkstep = 0

    do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, "nonnegative")

    cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
        vtkstep += 1
        minQ, maxQ = minimum(Q), maximum(Q)
        sumQ = weightedsum(Q)

        sumerror = (initialsumQ - sumQ) / initialsumQ

        @info @sprintf """Step  = %d
          minimum(Q)  = %.16e
          maximum(Q)  = %.16e
          sum error   = %.16e
          """ vtkstep minQ maxQ sumerror

        do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, "nonnegative")
    end

    callbacks = (cbTMAR, cbvtk)
    solve!(Q, odesolver; timeend = timeend, callbacks = callbacks)

    minQ, maxQ = minimum(Q), maximum(Q)
    finalsumQ = weightedsum(Q)
    sumerror = (initialsumQ - finalsumQ) / initialsumQ
    error = euclidean_distance(Q, Qe)

    @test minQ ≥ 0

    @info @sprintf """Finished
    minimum(Q) = %.16e
    maximum(Q) = %.16e
    L2 error   = %.16e
    sum error  = %.16e
    """ minQ maxQ error sumerror
end

let
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    FT = Float64
    dim = 2
    Ne = 20
    polynomialorder = 4

    problem = SwirlingFlow()

    brickrange = (
        range(FT(0); length = Ne + 1, stop = 1),
        range(FT(0); length = Ne + 1, stop = 1),
        range(FT(0); length = Ne + 1, stop = 1),
    )

    topology = BrickTopology(
        mpicomm,
        brickrange[1:dim],
        boundary = ntuple(d -> (3, 3), dim),
    )

    maxvelocity = 2
    elementsize = 1 / Ne
    dx = elementsize / polynomialorder^2
    CFL = 1
    dt = CFL * dx / maxvelocity

    vtkdir =
        abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk_nonnegative"))
    outputtime = 0.0625
    dt = outputtime / ceil(Int64, outputtime / dt)

    timeend = problem.period

    @info @sprintf """Starting
    FT               = %s
    dim              = %d
    Ne               = %d
    polynomial order = %d
    final time       = %.16e
    time step        = %.16e
    """ FT dim Ne polynomialorder timeend dt

    run(
        mpicomm,
        ArrayType,
        topology,
        problem,
        dt,
        polynomialorder,
        timeend,
        FT,
        vtkdir,
        outputtime,
    )
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

