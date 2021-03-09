"""
    ODESolvers

Ordinary differential equation solvers
"""
module ODESolvers

using LinearAlgebra
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using StaticArrays
using ..SystemSolvers
using ..MPIStateArrays: array_device, realview
using ..GenericCallbacks

export solve!, updatedt!, gettime, getsteps

abstract type AbstractODESolver end

"""
    gettime(solver::AbstractODESolver)

Returns the current simulation time of the ODE solver `solver`
"""
gettime(solver::AbstractODESolver) = solver.t

"""
    getdt(solver::AbstractODESolver)

Returns the current simulation time step of the ODE solver `solver`
"""
getdt(solver::AbstractODESolver) = solver.dt

"""
    getsteps(solver::AbstractODESolver)

Returns the number of completed time steps of the ODE solver `solver`
"""
getsteps(solver::AbstractODESolver) = solver.steps

"""
    ODESolvers.general_dostep!(Q, solver::AbstractODESolver, p,
                               timeend::Real, adjustfinalstep::Bool)

Use the solver to step `Q` forward in time from the current time, to the time
`timeend`. If `adjustfinalstep == true` then `dt` is adjusted so that the step
does not take the solution beyond the `timeend`.
"""
function general_dostep!(
    Q,
    solver::AbstractODESolver,
    p,
    timeend::Real;
    adjustfinalstep::Bool,
)
    time, dt = gettime(solver), getdt(solver)
    final_step = false
    if adjustfinalstep && time + dt > timeend
        orig_dt = dt
        dt = timeend - time
        updatedt!(solver, dt)
        final_step = true
    end
    @assert dt > 0

    dostep!(Q, solver, p, time)

    if !final_step
        updatetime!(solver, time + dt)
    else
        updatedt!(solver, orig_dt)
        updatetime!(solver, timeend)
    end
end

"""
    updatetime!(solver::AbstractODESolver, time)

Change the current time to `time` for the ODE solver `solver`.
"""
updatetime!(solver::AbstractODESolver, time) = (solver.t = time)

"""
    updatedt!(solver::AbstractODESolver, dt)

Change the time step size to `dt` for the ODE solver `solver`.
"""
updatedt!(solver::AbstractODESolver, dt) = (solver.dt = dt)

"""
    updatesteps!(solver::AbstractODESolver, dt)

Set the number of elapsed time steps for the ODE solver `solver`.
"""
updatesteps!(solver::AbstractODESolver, steps) = (solver.steps = steps)

isadjustable(solver::AbstractODESolver) = true

# {{{ run!
"""
    solve!(Q, solver::AbstractODESolver; timeend,
           stopaftertimeend=true, numberofsteps, callbacks)

Solves an ODE using the `solver` starting from a state `Q`. The state `Q` is
updated inplace. The final time `timeend` or `numberofsteps` must be specified.

A series of optional callback functions can be specified using the tuple
`callbacks`; see the `GenericCallbacks` module.
"""
function solve!(
    Q,
    solver::AbstractODESolver,
    param = nothing;
    timeend::Real = Inf,
    adjustfinalstep = true,
    numberofsteps::Integer = 0,
    callbacks = (),
    do_callback_init = true,
    do_callback_fini = true,
)

    @assert isfinite(timeend) || numberofsteps > 0
    if adjustfinalstep && !isadjustable(solver)
        error("$solver does not support time step adjustments. Can only be used with `adjustfinalstep=false`.")
    end
    t0 = gettime(solver)

    # Loop through an initialize callbacks (if they need it)
    if do_callback_init
        GenericCallbacks.init!(callbacks, solver, Q, param, t0)
    end

    step = 0
    time = t0
    while time < timeend
        step += 1
        updatesteps!(solver, step)

        time = general_dostep!(
            Q,
            solver,
            param,
            timeend;
            adjustfinalstep = adjustfinalstep,
        )

        val = GenericCallbacks.call!(callbacks, solver, Q, param, time)
        if val !== nothing && val > 0
            return gettime(solver)
        end

        # Figure out if we should stop
        if step == numberofsteps
            break
        end
    end

    # Loop through to fini callbacks
    if do_callback_fini
        GenericCallbacks.fini!(callbacks, solver, Q, param, time)
    end

    return gettime(solver)
end
# }}}

include("BackwardEulerSolvers.jl")
include("MultirateInfinitesimalGARKExplicit.jl")
include("MultirateInfinitesimalGARKDecoupledImplicit.jl")
include("MultirateInfinitesimalStepMethod.jl")
include("LowStorageRungeKuttaMethod.jl")
include("LowStorageRungeKutta3NMethod.jl")
include("StrongStabilityPreservingRungeKuttaMethod.jl")
include("AdditiveRungeKuttaMethod.jl")
include("MultirateRungeKuttaMethod.jl")
include("SplitExplicitMethod.jl")
include("DifferentialEquations.jl")
include("CplSolver.jl")

end # module
