using Test
using ClimateMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.BalanceLaws
using ClimateMachine.Ocean
using ClimateMachine.Ocean.SplitExplicit01
using ClimateMachine.Ocean.OceanProblems

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function config_simple_box(
    name,
    resolution,
    dimensions,
    boundary_conditions;
    dt_slow = 90.0 * 60.0,
    dt_fast = 240.0,
)

    problem = OceanGyre{FT}(
        dimensions...;
        τₒ = 0.2,
        λʳ = 10 // 86400,
        θᴱ = 10,
        BC = boundary_conditions,
    )

    add_fast_substeps = 2
    numImplSteps = 5
    numImplSteps > 0 ? ivdc_dt = dt_slow / FT(numImplSteps) : ivdc_dt = dt_slow
    model_3D = OceanModel{FT}(
        param_set,
        problem;
        cʰ = 1,
        κᶜ = FT(0.1),
        add_fast_substeps = add_fast_substeps,
        numImplSteps = numImplSteps,
        ivdc_dt = ivdc_dt,
    )

    N, Nˣ, Nʸ, Nᶻ = resolution
    resolution = (Nˣ, Nʸ, Nᶻ)

    config = ClimateMachine.OceanSplitExplicitConfiguration(
        name,
        N,
        resolution,
        param_set,
        model_3D;
        solver_type = SplitExplicitSolverType{FT}(dt_slow, dt_fast),
    )

    return config
end

function run_simple_box(driver_config, timespan; refDat = ())

    timestart, timeend = timespan
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_dt = driver_config.solver_type.dt_slow,
    )

    ## Create a callback to report state statistics for main MPIStateArrays
    ## every ntFreq timesteps.
    nt_freq = 1 # floor(Int, 1 // 10 * solver_config.timeend / solver_config.dt)
    cb = ClimateMachine.StateCheck.sccreate(
        [
            (solver_config.Q, "3D state"),
            (solver_config.dg.state_auxiliary, "3D aux"),
            (solver_config.dg.modeldata.Q_2D, "2D state"),
            (solver_config.dg.modeldata.dg_2D.state_auxiliary, "2D aux"),
        ],
        nt_freq;
        prec = 12,
    )

    result = ClimateMachine.invoke!(solver_config; user_callbacks = [cb])

    ## Check results against reference if present
    ClimateMachine.StateCheck.scprintref(cb)
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cb, refDat)
    end
end
