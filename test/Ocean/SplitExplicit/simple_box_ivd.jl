#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.BalanceLaws: vars_state, Prognostic, Auxiliary
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates: flattenednames
using ClimateMachine.Ocean.SplitExplicit01
using ClimateMachine.Ocean.OceanProblems
using ClimateMachine.GenericCallbacks
using ClimateMachine.VTK

using Test
using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

const ArrayType = ClimateMachine.array_type()

function main(BC)
    mpicomm = MPI.COMM_WORLD

    brickrange_2D = (xrange, yrange)
    topl_2D =
        BrickTopology(mpicomm, brickrange_2D, periodicity = (false, false))
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    brickrange_3D = (xrange, yrange, zrange)
    topl_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = (false, false, false),
        boundary = ((1, 1), (1, 1), (2, 3)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob = OceanGyre{FT}(Lˣ, Lʸ, H; τₒ = τₒ, λʳ = λʳ, θᴱ = θᴱ, BC = BC)
    gravity::FT = grav(param_set)

    #- set model time-step:
    dt_fast = 240
    dt_slow = 5400

    nout = ceil(Int64, tout / dt_slow)
    dt_slow = tout / nout
    numImplSteps > 0 ? ivdc_dt = dt_slow / FT(numImplSteps) : ivdc_dt = dt_slow

    model = OceanModel{FT}(
        param_set,
        prob,
        cʰ = cʰ,
        add_fast_substeps = add_fast_substeps,
        numImplSteps = numImplSteps,
        ivdc_dt = ivdc_dt,
        κᶜ = FT(0.1),
    )
    # model = OceanModel{FT}(prob, cʰ = cʰ, fₒ = FT(0), β = FT(0) )	
    # model = OceanModel{FT}(prob, cʰ = cʰ, νʰ = FT(1e3), νᶻ = FT(1e-3) )	
    # model = OceanModel{FT}(prob, cʰ = cʰ, νʰ = FT(0), fₒ = FT(0), β = FT(0) )

    barotropicmodel = BarotropicModel(model)

    minΔx = min_node_distance(grid_3D, HorizontalDirection())
    minΔz = min_node_distance(grid_3D, VerticalDirection())
    #- 2 horiz directions
    gravity_max_dT = 1 / (2 * sqrt(gravity * H) / minΔx)
    # dt_fast = minimum([gravity_max_dT])

    #- 2 horiz directions + harmonic visc or diffusion: 2^2 factor in CFL:
    viscous_max_dT = 1 / (2 * model.νʰ / minΔx^2 + model.νᶻ / minΔz^2) / 4
    diffusive_max_dT = 1 / (2 * model.κʰ / minΔx^2 + model.κᶻ / minΔz^2) / 4
    # dt_slow = minimum([diffusive_max_dT, viscous_max_dT])

    @info @sprintf(
        """Update
           Gravity Max-dT = %.1f
           Timestep       = %.1f""",
        gravity_max_dT,
        dt_fast
    )

    @info @sprintf(
        """Update
       Viscous   Max-dT = %.1f
       Diffusive Max-dT = %.1f
       Timestep      = %.1f""",
        viscous_max_dT,
        diffusive_max_dT,
        dt_slow
    )

    barotropic_dg = DGModel(
        barotropicmodel,
        grid_2D,
        # CentralNumericalFluxFirstOrder(),
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_2D = init_ode_state(barotropic_dg, FT(0); init_on_cpu = true)

    dg = OceanDGModel(
        model,
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        modeldata = (barotropic_dg, Q_2D),
    )

    Q_3D = init_ode_state(dg, FT(0); init_on_cpu = true)

    lsrk_ocean = LSRK54CarpenterKennedy(dg, Q_3D, dt = dt_slow, t0 = 0)
    lsrk_barotropic =
        LSRK54CarpenterKennedy(barotropic_dg, Q_2D, dt = dt_fast, t0 = 0)

    odesolver = SplitExplicitLSRK2nSolver(lsrk_ocean, lsrk_barotropic)

    #-- Set up State Check call back for config state arrays, called every ntFreq time steps
    ntFreq = 1
    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [
            (Q_3D, "3D state"),
            (dg.state_auxiliary, "3D aux"),
            (Q_2D, "2D state"),
            (barotropic_dg.state_auxiliary, "2D aux"),
            # (dg.diffstate,"oce diff",),	
            # (lsrk_ocean.dQ,"oce_dQ",),	
            # (dg.modeldata.tendency_dg.state_auxiliary,"tend Int aux",),	
            # (dg.modeldata.conti3d_Q,"conti3d_Q",),
            # (barotropic_dg.diffstate,"baro diff",),	
            # (lsrk_barotropic.dQ,"baro_dQ",)
        ],
        ntFreq;
        prec = 12,
    )

    step = [0, 0]
    cbvector = make_callbacks(
        vtkpath,
        step,
        nout,
        mpicomm,
        odesolver,
        dg,
        model,
        Q_3D,
        barotropic_dg,
        barotropicmodel,
        Q_2D,
    )

    eng0 = norm(Q_3D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    cbv = (cbvector..., cbcs_dg)
    solve!(Q_3D, odesolver; timeend = timeend, callbacks = cbv)

    ## Enable the code block below to print table for use in reference value code
    ## reference value code sits in a file named $(@__FILE__)_refvals.jl. It is hand
    ## edited using code generated by block below when reference values are updated.
    regenRefVals = false
    if regenRefVals
        ## Print state statistics in format for use as reference values
        println(
            "# SC ========== Test number ",
            1,
            " reference values and precision match template. =======",
        )
        println("# SC ========== $(@__FILE__) test reference values ======================================")
        ClimateMachine.StateCheck.scprintref(cbcs_dg)
        println("# SC ====================================================================================")
    end

    ## Check results against reference if present
    checkRefVals = true
    if checkRefVals
        include("../refvals/simple_box_ivd_refvals.jl")
        refDat = (refVals[1], refPrecs[1])
        checkPass = ClimateMachine.StateCheck.scdocheck(cbcs_dg, refDat)
        checkPass ? checkRep = "Pass" : checkRep = "Fail"
        @test checkPass
        @info @sprintf("""Compare vs RefVals: %s""", checkRep)
    end

    return nothing
end

function make_callbacks(
    vtkpath,
    step,
    nout,
    mpicomm,
    odesolver,
    dg_slow,
    model_slow,
    Q_slow,
    dg_fast,
    model_fast,
    Q_fast,
)

    if isdir(vtkpath)
        rm(vtkpath, recursive = true)
    end
    mkpath(vtkpath)
    mkpath(vtkpath * "/slow")
    mkpath(vtkpath * "/fast")

    function do_output(span, step, model, dg, Q)
        outprefix = @sprintf(
            "%s/%s/mpirank%04d_step%04d",
            vtkpath,
            span,
            MPI.Comm_rank(mpicomm),
            step
        )
        @info "doing VTK output" outprefix
        statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
        auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)

        mycomm = Q.mpicomm
        ## Generate the pvtu file for these vtk files
        if MPI.Comm_rank(mpicomm) == 0 && MPI.Comm_size(mpicomm) > 1
            ## name of the pvtu file
            pvtuprefix = @sprintf("%s/%s/step%04d", vtkpath, span, step)
            ## name of each of the ranks vtk files
            prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                @sprintf("mpirank%04d_step%04d", i - 1, step)
            end
            writepvtu(
                pvtuprefix,
                prefixes,
                (statenames..., auxnames...),
                eltype(Q),
            )
            @info "Done writing VTK: $pvtuprefix"
        end

    end

    do_output("slow", step[1], model_slow, dg_slow, Q_slow)
    cbvtk_slow = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("slow", step[1], model_slow, dg_slow, Q_slow)
        step[1] += 1
        nothing
    end

    do_output("fast", step[2], model_fast, dg_fast, Q_fast)
    cbvtk_fast = GenericCallbacks.EveryXSimulationSteps(nout) do (init = false)
        do_output("fast", step[2], model_fast, dg_fast, Q_fast)
        step[2] += 1
        nothing
    end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q_slow)
            @info @sprintf(
                """Update
                simtime = %8.2f / %8.2f
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(odesolver),
                timeend,
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end

    # return (cbvtk_slow, cbvtk_fast, cbinfo)
    return (cbinfo,)
end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath =
    abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk_simple_box_ivd"))

const timeend = 5 * 24 * 3600 # s
const tout = 24 * 3600 # s

const N = 4
const Nˣ = 20
const Nʸ = 20
const Nᶻ = 20
const Lˣ = 4e6  # m
const Lʸ = 4e6  # m
const H = 1000  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

#const cʰ = sqrt(gravity * H)
const cʰ = 1  # typical of ocean internal-wave speed
const cᶻ = 0

#- inverse ratio of additional fast time steps (for weighted average)
#  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
# e.g., = 1 --> 100% more ; = 2 --> 50% more ; = 3 --> 33% more ...
add_fast_substeps = 2

#- number of Implicit vertical-diffusion sub-time-steps within one model full time-step
# default = 0 : disable implicit vertical diffusion
numImplSteps = 5

const τₒ = 2e-1  # (Pa = N/m^2)
const λʳ = 10 // 86400 # m/s
const θᴱ = 10    # deg.C

BC = (
    ClimateMachine.Ocean.SplitExplicit01.CoastlineNoSlip(),
    ClimateMachine.Ocean.SplitExplicit01.OceanFloorNoSlip(),
    ClimateMachine.Ocean.SplitExplicit01.OceanSurfaceStressForcing(),
)

@testset "$(@__FILE__)" begin
    main(BC)
end
