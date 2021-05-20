# ClimateMachine solver configurations
#
# Contains helper functions to establish solver configurations to be
# used with the ClimateMachine driver.

"""
    ClimateMachine.SolverConfiguration

Parameters needed by `ClimateMachine.solve!()` to run a simulation.
"""
mutable struct SolverConfiguration{FT}
    name::String
    mpicomm::MPI.Comm
    param_set::AbstractParameterSet
    dg::SpaceDiscretization
    Q::MPIStateArray
    t0::FT
    timeend::FT
    dt::FT
    init_on_cpu::Bool
    numberofsteps::Int
    init_args::Any
    solver::Any
    ode_solver_type::Any
    diffdir::Any
    CFL::FT
end

"""
    DGMethods.courant(local_cfl, solver_config::SolverConfiguration;
                      Q=solver_config.Q, dt=solver_config.dt)

Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain with the model defined by `solver_config`. The
keyword arguments `Q` and `dt` can be used to call the courant method with a
different state `Q` or time step `dt` than are defined in `solver_config`.
"""
DGMethods.courant(
    f,
    sc::SolverConfiguration;
    Q = sc.Q,
    dt = sc.dt,
    simtime = gettime(sc.solver),
    direction = EveryDirection(),
) = DGMethods.courant(f, sc.dg, sc.dg.balance_law, Q, dt, simtime, direction)

get_direction(::ClimateMachineConfigType) = EveryDirection()
get_direction(::SingleStackConfigType) = VerticalDirection()

get_solver_type(::Type{FT}, ::AtmosGCMConfigType) where {FT} =
    DefaultSolverType()
get_solver_type(::Type{FT}, ::AtmosLESConfigType) where {FT} =
    solver_type = IMEXSolverType(
        implicit_solver = SingleColumnLU,
        implicit_solver_adjustable = false,
    )
get_solver_type(::Type{FT}, ::OceanSplitExplicitConfigType) where {FT} =
    SplitExplicitSolverType{FT}(90.0 * 60.0, 240.0)
get_solver_type(::Type{FT}, ::OceanBoxGCMConfigType) where {FT} =
    ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)
get_solver_type(::Type{FT}, ::SingleStackConfigType) where {FT} =
    ExplicitSolverType()
get_solver_type(::Type{FT}, ::MultiColumnLandConfigType) where {FT} =
    ExplicitSolverType()

"""
    ClimateMachine.SolverConfiguration(
        t0::FT,
        timeend::FT,
        driver_config::DriverConfiguration,
        init_args...;
        init_on_cpu=false,
        ode_solver_type,
        ode_dt=nothing,
        modeldata=nothing,
        Courant_number=0.4,
        diffdir=EveryDirection(),
    )

Set up the DG model per the specified driver configuration, set up
the ODE solver, and return a `SolverConfiguration` to be used with
`ClimateMachine.invoke!()`.

# Arguments:
# - `t0::FT`: simulation start time.
# - `timeend::FT`: simulation end time.
# - `driver_config::DriverConfiguration`: from `AtmosLESConfiguration()`, etc.
# - `init_args...`: passed through to `init_state_prognostic!()`.
# - `init_on_cpu=false`: run `init_state_prognostic!()` on CPU?
# - `ode_solver_type`: override solver choice.
# - `ode_dt=nothing`: override timestep computation.
# - `modeldata=nothing`: passed through to `DGModel`.
# - `Courant_number=0.4`: for the simulation.
# - `diffdir=EveryDirection()`: restrict diffusivity direction.
# - `direction=EveryDirection()`: restrict diffusivity direction.
# - `timeend_dt_adjust=true`: should `dt` be adjusted to hit `timeend` exactly
# - `CFL_direction=EveryDirection()`: direction for `calculate_dt`
# - `sim_time`: run for the specified time (in simulation seconds).
# - `fixed_number_of_steps`: if `≥0` perform specified number of steps.

Note that `diffdir`, `direction`, and `CFL_direction` are `VerticalDirection()`
when `driver_config.config_type isa SingleStackConfigType`.
"""
function SolverConfiguration(
    t0::FT,
    timeend::FT,
    driver_config::DriverConfiguration,
    init_args...;
    init_on_cpu = false,
    ode_solver_type = get_solver_type(FT, driver_config.config_type),
    ode_dt = nothing,
    modeldata = nothing,
    Courant_number = nothing,
    diffdir = get_direction(driver_config.config_type),
    direction = get_direction(driver_config.config_type),
    timeend_dt_adjust = true,
    CFL_direction = get_direction(driver_config.config_type),
    sim_time = Settings.sim_time,
    fixed_number_of_steps = Settings.fixed_number_of_steps,
    skip_update_aux = false,
) where {FT <: AbstractFloat}
    @tic SolverConfiguration

    bl = driver_config.bl
    grid = driver_config.grid

    # Create the DG model and initialize the ODE state. If we're restarting,
    # use state data from the checkpoint.
    if Settings.restart_from_num > 0
        s_Q, s_aux, t0 = read_checkpoint(
            Settings.checkpoint_dir,
            driver_config.name,
            driver_config.array_type,
            driver_config.mpicomm,
            Settings.restart_from_num,
        )

        state_auxiliary = restart_auxiliary_state(bl, grid, s_aux, direction)

        if hasproperty(driver_config.config_info, :dg)
            dg = driver_config.config_info.dg

            dg.state_auxiliary .= state_auxiliary
        else
            dg = SpaceDiscretization(
                driver_config;
                state_auxiliary = state_auxiliary,
                direction = direction,
                diffusion_direction = diffdir,
                modeldata = modeldata,
                check_for_crashes = Settings.checkpoint_on_crash,
            )
        end

        @info @sprintf(
            "Initializing %s from time %8.2f",
            driver_config.name,
            t0
        )
        Q = restart_ode_state(dg, s_Q; init_on_cpu = init_on_cpu)
    else
        if hasproperty(driver_config.config_info, :dg)
            dg = driver_config.config_info.dg
        else
            dg = SpaceDiscretization(
                driver_config;
                fill_nan = Settings.debug_init,
                direction = direction,
                diffusion_direction = diffdir,
                modeldata = modeldata,
                check_for_crashes = Settings.checkpoint_on_crash,
            )
        end

        if Settings.debug_init
            write_debug_init_vtk_and_pvtu(
                "init_auxiliary",
                driver_config,
                dg,
                dg.state_auxiliary,
                flattenednames(vars_state(bl, Auxiliary(), FT)),
            )
        end

        @info @sprintf("Initializing %s", driver_config.name,)
        Q = init_ode_state(dg, FT(0), init_args...; init_on_cpu = init_on_cpu)
        if driver_config.filter !== nothing
            Filters.apply!(Q, :, dg.grid, driver_config.filter)
        end

        if Settings.debug_init
            write_debug_init_vtk_and_pvtu(
                "init_prognostic",
                driver_config,
                dg,
                Q,
                flattenednames(vars_state(bl, Prognostic(), FT)),
            )
        end
    end
    if !skip_update_aux
        update_auxiliary_state!(dg, bl, Q, FT(0), dg.grid.topology.realelems)
    end

    if Settings.debug_init
        write_debug_init_vtk_and_pvtu(
            "first_update_auxiliary",
            driver_config,
            dg,
            dg.state_auxiliary,
            flattenednames(vars_state(bl, Auxiliary(), FT)),
        )
    end

    # default Courant number
    # TODO: Think about revising this or drop it entirely.
    # This is difficult to determine/approximate
    # for MIS and general multirate methods.
    if Courant_number === nothing
        if isa(ode_solver_type, ExplicitSolverType)
            if ode_solver_type.solver_method == LSRK144NiegemannDiehlBusch
                Courant_number = FT(1.7)
            else
                @warn "Courant_number configured for ode_solver_type.solver_method == LSRK54CarpenterKennedy"
                Courant_number = FT(0.3)
            end
        else
            Courant_number = FT(0.5)
        end
    end

    # initial Δt specified or computed
    if ode_dt === nothing
        dtmodel = getdtmodel(ode_solver_type, bl)
        ode_dt = ClimateMachine.DGMethods.calculate_dt(
            dg,
            dtmodel,
            Q,
            Courant_number,
            t0,
            CFL_direction,
        )
    end
    if !isnan(sim_time)
        timeend = sim_time
    end
    if fixed_number_of_steps < 0
        numberofsteps = convert(Int, cld(timeend - t0, ode_dt))
        timeend_dt_adjust && (ode_dt = (timeend - t0) / numberofsteps)
    else
        numberofsteps = fixed_number_of_steps
        timeend = fixed_number_of_steps * ode_dt
    end

    # create the solver
    solver = solversetup(ode_solver_type, dg, Q, ode_dt, t0, diffdir)

    @toc SolverConfiguration

    return SolverConfiguration(
        driver_config.name,
        driver_config.mpicomm,
        driver_config.param_set,
        dg,
        Q,
        t0,
        timeend,
        ode_dt,
        init_on_cpu,
        numberofsteps,
        init_args,
        solver,
        ode_solver_type,
        diffdir,
        Courant_number,
    )
end

function write_debug_init_vtk_and_pvtu(
    suffix_name,
    driver_config,
    dg,
    state,
    state_names,
)
    mpicomm = driver_config.mpicomm
    bl = driver_config.bl

    vprefix = @sprintf(
        "%s_mpirank%04d_%s",
        driver_config.name,
        MPI.Comm_rank(mpicomm),
        suffix_name,
    )
    out_prefix = joinpath(Settings.output_dir, vprefix)

    writevtk(
        out_prefix,
        state,
        dg,
        state_names;
        number_sample_points = Settings.vtk_number_sample_points,
    )

    # Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        pprefix = @sprintf("%s_%s", driver_config.name, suffix_name)
        pvtuprefix = joinpath(Settings.output_dir, pprefix)

        # name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_%s", driver_config.name, i - 1, suffix_name,)
        end
        writepvtu(pvtuprefix, prefixes, state_names, eltype(state))
    end
end

include("solver_config_wrappers.jl")
