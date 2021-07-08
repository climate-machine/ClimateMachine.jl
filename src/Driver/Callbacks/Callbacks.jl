module Callbacks

using CUDA
using Dates
using KernelAbstractions
using LinearAlgebra
using MPI
using Printf
using Statistics

using CLIMAParameters
using CLIMAParameters.Planet: day

using ..ClimateMachine
using ..BalanceLaws
using ..Courant
using ..Checkpoint
using ..DGMethods
using ..BalanceLaws: vars_state, Prognostic, Auxiliary
using ..Diagnostics
using ..GenericCallbacks
using ..Mesh.Grids: EveryDirection, HorizontalDirection, VerticalDirection
using ..MPIStateArrays
using ..ODESolvers
using ..TicToc
using ..VariableTemplates
using ..VTK

_sync_device(::Type{CuArray}) = synchronize()
_sync_device(::Type{Array}) = nothing

"""
    SummaryLogCallback([stimeend])

Log a summary of the current run.

The optional `stimeend` argument is used to print the total simulation time.
"""
mutable struct SummaryLogCallback
    stimeend::Union{Nothing, Real}
    wtimestart::DateTime # wall time at start of simulation
    function SummaryLogCallback(stimeend = nothing)
        new(stimeend, now())
    end
end

function GenericCallbacks.init!(cb::SummaryLogCallback, solver, Q, param, t)
    cb.wtimestart = now()
    return nothing
end
function GenericCallbacks.call!(cb::SummaryLogCallback, solver, Q, param, t)
    wallclock_time_ms = Dates.now() - cb.wtimestart
    wallclock = Dates.format(
        convert(Dates.DateTime, wallclock_time_ms),
        Dates.dateformat"HH:MM:SS",
    )
    normQ = norm(Q)
    if cb.stimeend === nothing
        simtime = @sprintf "%8.2f" t
    else
        simtime = @sprintf "%8.2f / %8.2f" t cb.stimeend
    end
    wallclock_s = wallclock_time_ms.value / 1000
    efficiency = @sprintf "%8.4f" t / wallclock_s

    estimated_wallclock_end = cb.stimeend * wallclock_s / t

    estimated_wallclock_end_ms =
        Dates.Millisecond(ceil(Int, estimated_wallclock_end * 1000))

    estimated_remaining = Dates.format(
        convert(Dates.DateTime, estimated_wallclock_end_ms),
        Dates.dateformat"HH:MM:SS",
    )
    @info @sprintf(
        """
        Update
            simtime = %s
            wallclock = %s
            efficiency (simtime / wallclock) = %s
            wallclock end (estimated) = %s
            norm(Q) = %.16e""",
        simtime,
        wallclock,
        efficiency,
        estimated_remaining,
        normQ,
    )
    if !isfinite(normQ)
        show_not_finite_fields(Q)
        error("norm(Q) is not finite")
    end
    return nothing
end
function GenericCallbacks.fini!(cb::SummaryLogCallback, solver, Q, param, t)
    return nothing
end

"""
    show_updates(show_updates_opt, solver_config, user_info_callback)

Return a callback function that shows simulation updates at the specified
interval and also invokes the user-specified info callback.
"""
function show_updates(show_updates_opt, solver_config, user_info_callback)
    timeend = solver_config.timeend

    cb_constr = CB_constructor(show_updates_opt, solver_config)
    if cb_constr !== nothing
        return cb_constr((
            SummaryLogCallback(solver_config.timeend),
            GenericCallbacks.AtInit(user_info_callback),
        ))
    else
        return nothing
    end
end

"""
    diagnostics(diagnostics_opt, solver_config, dgn_starttime, diagnostics_config)

Return callback functions that shows simulation updates at the specified
interval and also invokes the user-specified info callback.
"""
function diagnostics(
    diagnostics_opt,
    solver_config,
    dgn_starttime,
    diagnostics_config,
)
    if diagnostics_opt != "never" && diagnostics_config !== nothing
        # set up a callback for each diagnostics group
        dgncbs = ()
        for dgngrp in diagnostics_config.groups
            cb_constr =
                CB_constructor(diagnostics_opt, solver_config, dgngrp.interval)
            cb_constr === nothing && continue
            dgncbs = (dgncbs..., cb_constr(dgngrp))
        end
        return dgncbs
    else
        return nothing
    end
end

const minimum_vtk_interval = 0.01

"""
    vtk(vtk_opt, solver_config)

Return a callback that saves state and auxiliary variables to a VTK
file.
"""
function vtk(vtk_opt, solver_config, output_dir, number_sample_points)
    cb_constr = CB_constructor(vtk_opt, solver_config)
    cb_constr === nothing && return nothing

    vtknum = Ref(1)
    last_collect_time = Ref(0.0)

    mpicomm = solver_config.mpicomm
    dg = solver_config.dg
    bl = dg.balance_law
    Q = solver_config.Q
    FT = eltype(Q)

    cb_vtk = GenericCallbacks.AtInitAndFini() do
        sim_time = ODESolvers.gettime(solver_config.solver)
        if sim_time - last_collect_time[] < minimum_vtk_interval
            return nothing
        end

        # TODO: make an object
        vprefix = @sprintf(
            "%s_mpirank%04d_num%04d",
            solver_config.name,
            MPI.Comm_rank(mpicomm),
            vtknum[],
        )
        outprefix = joinpath(output_dir, vprefix)

        statenames = flattenednames(vars_state(bl, Prognostic(), FT))
        auxnames = flattenednames(vars_state(bl, Auxiliary(), FT))

        writevtk(
            outprefix,
            Q,
            dg,
            statenames,
            dg.state_auxiliary,
            auxnames;
            number_sample_points = number_sample_points,
        )

        # Generate the pvtu file for these vtk files
        if MPI.Comm_rank(mpicomm) == 0
            # name of the pvtu file
            pprefix = @sprintf("%s_num%04d", solver_config.name, vtknum[])
            pvtuprefix = joinpath(output_dir, pprefix)

            # name of each of the ranks vtk files
            prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                @sprintf(
                    "%s_mpirank%04d_num%04d",
                    solver_config.name,
                    i - 1,
                    vtknum[],
                )
            end
            writepvtu(
                pvtuprefix,
                prefixes,
                (statenames..., auxnames...),
                eltype(Q),
            )
        end

        vtknum[] += 1
        last_collect_time[] = sim_time
        nothing
    end
    return cb_constr(cb_vtk)
end

"""
    monitor_timestep_duration(mtd_opt, array_type, comm)

Returns a callback function that displays wall-clock time per time-step
statistics across MPI ranks in the communicator `comm`.  The times are
averaged over the `mtd_opt` time steps requested for output.  The
`array_type` is used to synchronize the compute device.
"""
function monitor_timestep_duration(mtd_opt, array_type, comm)
    if mtd_opt == "never"
        return nothing
    elseif !endswith(mtd_opt, "steps")
        @warn @sprintf(
            """
            monitor-timestep-duration must be in 'steps'; %s unrecognized; disabling""",
            mtd_opt,
        )
        return nothing
    end
    steps = parse(Int, mtd_opt[1:(end - 5)])

    _sync_device(array_type)
    before = time_ns()

    cb_mtd = GenericCallbacks.EveryXSimulationSteps(steps) do
        _sync_device(array_type)
        after = time_ns()

        time_per_timesteps = after - before

        times = MPI.Gather(time_per_timesteps, 0, comm)
        if MPI.Comm_rank(comm) == 0
            ns_per_s = 1e9
            times = times ./ ns_per_s ./ steps

            @info @sprintf(
                """
                Wall-clock time per time-step (statistics across MPI ranks)
                   maximum (s) = %25.16e
                   minimum (s) = %25.16e
                   median  (s) = %25.16e
                   std     (s) = %25.16e""",
                maximum(times),
                minimum(times),
                median(times),
                std(times),
            )
        end

        _sync_device(array_type)
        before = time_ns()

        nothing
    end

    return cb_mtd
end

"""
    monitor_courant_numbers(mcn_opt, solver_config)

Return a callback function that displays Courant numbers for the simulation
at `mcn_opt` intervals.
"""
function monitor_courant_numbers(mcn_opt, solver_config)
    cb_constr = CB_constructor(mcn_opt, solver_config)
    cb_constr === nothing && return nothing

    cb_cfl = cb_constr() do
        Δt = solver_config.dt
        c_v = courant(
            nondiffusive_courant,
            solver_config,
            direction = VerticalDirection(),
        )
        c_h = courant(
            nondiffusive_courant,
            solver_config,
            direction = HorizontalDirection(),
        )
        ca_v = courant(
            advective_courant,
            solver_config,
            direction = VerticalDirection(),
        )
        ca_h = courant(
            advective_courant,
            solver_config,
            direction = HorizontalDirection(),
        )
        cd_v = courant(
            diffusive_courant,
            solver_config,
            direction = VerticalDirection(),
        )
        cd_h = courant(
            diffusive_courant,
            solver_config,
            direction = HorizontalDirection(),
        )
        simtime = ODESolvers.gettime(solver_config.solver)
        @info @sprintf(
            """
            Courant numbers at simtime: %8.2f, Δt = %8.2f s
                Acoustic (vertical) Courant number    = %.2g
                Acoustic (horizontal) Courant number  = %.2g
                Advection (vertical) Courant number   = %.2g
                Advection (horizontal) Courant number = %.2g
                Diffusion (vertical) Courant number   = %.2g
                Diffusion (horizontal) Courant number = %.2g""",
            simtime,
            Δt,
            c_v,
            c_h,
            ca_v,
            ca_h,
            cd_v,
            cd_h,
        )
        nothing
    end
    return cb_cfl
end

function adapt_timestep(adp_opt, solver_config)
    cb_constr = CB_constructor(adp_opt, solver_config)
    cb_constr === nothing && return nothing
    cb_adp = cb_constr() do
        dt = solver_config.solver.dt
        dg = solver_config.dg
        bl = dg.balance_law
        Q = solver_config.Q
        t0 = solver_config.t0
        ode_solver_type = solver_config.ode_solver_type
        dtmodel = getdtmodel(ode_solver_type, bl)
        ndt = ClimateMachine.DGMethods.calculate_dt(
            dg,
            dtmodel,
            Q,
            solver_config.CFL,
            t0,
            solver_config.diffdir,
        )
        @info @sprintf("""Updating time step: %8.16f => %8.16f""", dt, ndt)
        updatedt!(solver_config.solver, ndt)
        nothing
    end
    return cb_adp
end

"""
    checkpoint(
        checkpoint_opt,
        checkpoint_keep_one,
        solver_config,
        checkpoint_dir,
    )

Return a callback function that runs at `checkpoint_opt` intervals
and stores a simulation checkpoint into `checkpoint_dir` identified
by a running number.
"""
function checkpoint(
    checkpoint_opt,
    checkpoint_keep_one,
    solver_config,
    checkpoint_dir,
)
    cb_constr = CB_constructor(checkpoint_opt, solver_config)
    cb_constr === nothing && return nothing

    cpnum = Ref(1)
    cb_checkpoint = cb_constr() do
        write_checkpoint(
            solver_config,
            checkpoint_dir,
            solver_config.name,
            solver_config.mpicomm,
            cpnum[],
        )
        if checkpoint_keep_one
            rm_checkpoint(
                checkpoint_dir,
                solver_config.name,
                solver_config.mpicomm,
                cpnum[] - 1,
            )
        end
        cpnum[] += 1
        nothing
    end
    return cb_checkpoint
end

"""
    ConsCallback(dg, mass, energy, show_cons)

Check mass and energy conservation against specified tolerances.
"""
mutable struct ConsCallback{FT}
    bl::BalanceLaw
    varname::String
    error_threshold::FT
    show::Bool
    Σvar₀::FT
end

function GenericCallbacks.init!(cb::ConsCallback, solver, Q, param, t)
    FT = eltype(Q)
    idx = varsindices(vars_state(cb.bl, Prognostic(), FT), cb.varname)
    cb.Σvar₀ = weightedsum(Q, idx)
    return nothing
end
function GenericCallbacks.call!(cb::ConsCallback, solver, Q, param, t)
    FT = eltype(Q)
    idx = varsindices(vars_state(cb.bl, Prognostic(), FT), cb.varname)
    Σvar = weightedsum(Q, idx)
    δvar = (Σvar - cb.Σvar₀) / cb.Σvar₀

    if abs(δvar) > cb.error_threshold
        error("abs(δ$(cb.varname)) > $(cb.error_threshold)")
    end

    if cb.show
        simtime = @sprintf "%8.2f" t
        @info @sprintf(
            """
            Conservation
                simtime = %s
                abs(δ%s) = %.5e""",
            simtime,
            cb.varname,
            abs(δvar),
        )
    end
    return nothing
end
function GenericCallbacks.fini!(cb::ConsCallback, solver, Q, param, t)
    return nothing
end

"""
    check_cons(
        check_cons,
        solver_config,
    )

Return a callback function for each element of `check_cons`, which must be
a tuple of [`ConservationCheck`s](@ref ClimateMachine.ConservationCheck).
"""
function check_cons(check_cons, solver_config)
    cbs = ()
    for cc in check_cons
        cb_constr = CB_constructor(cc.interval, solver_config)
        cb_constr === nothing && continue

        FT = eltype(solver_config.Q)
        cb = cb_constr((ConsCallback(
            solver_config.dg.balance_law,
            cc.varname,
            FT(cc.error_threshold),
            cc.show,
            FT(0),
        ),))

        cbs = (cbs..., cb)
    end

    return cbs
end

"""
    CB_constructor(interval, solver_config, default)

Parse the specified `interval` and return the appropriate `GenericCallbacks`
constructor. If `interval` is "default", then use `default` as the interval.
"""
function CB_constructor(interval::String, solver_config, default = nothing)
    mpicomm = solver_config.mpicomm
    solver = solver_config.solver
    dg = solver_config.dg
    bl = dg.balance_law
    param_set = parameter_set(bl)
    secs_per_day = day(param_set)

    if (
        m = match(r"^([0-9\.]+)(smonths|sdays|shours|smins|ssecs)$", interval)
    ) !== nothing
        n = parse(Float64, m[1])
        if m[2] == "smonths"
            ssecs = 30 * secs_per_day * n
        elseif m[2] == "sdays"
            ssecs = secs_per_day * n
        elseif m[2] == "shours"
            ssecs = 60 * 60 * n
        elseif m[2] == "smins"
            ssecs = 60 * n
        elseif m[2] == "ssecs"
            ssecs = n
        end
        return cb -> GenericCallbacks.EveryXSimulationTime(cb, ssecs)
    elseif (m = match(r"^([0-9]+)(steps)$", interval)) !== nothing
        steps = parse(Int, m[1])
        return cb -> GenericCallbacks.EveryXSimulationSteps(cb, steps)
    elseif (m = match(r"^([0-9\.]+)(hours|mins|secs)$", interval)) !== nothing
        n = parse(Float64, m[1])
        if m[2] == "hours"
            secs = 60 * 60 * n
        elseif m[2] == "mins"
            secs = 60 * n
        elseif m[2] == "secs"
            secs = n
        end
        return cb -> GenericCallbacks.EveryXWallTimeSeconds(cb, secs, mpicomm)
    elseif interval == "default"
        if default === nothing
            @warn "no default available; ignoring"
            return nothing
        elseif default === ""
            return nothing
        else
            return CB_constructor(default, solver_config, "")
        end
    elseif interval == "never"
        return nothing
    else
        @warn @sprintf(
            """
%s: unrecognized interval; ignoring""",
            interval,
        )
        return nothing
    end
end

function __init__()
    tictoc()
end

end # module
