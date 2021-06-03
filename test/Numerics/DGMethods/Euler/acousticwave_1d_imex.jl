using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, equiangular_cubed_sphere_warp, grid1d
using ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid, VerticalDirection, min_node_distance
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: DGModel, init_ode_state, remainder_DGModel
using ClimateMachine.DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics:
    air_density, soundspeed_air, internal_energy, PhaseDry_pT, PhasePartition
using ClimateMachine.TemperatureProfiles: IsothermalProfile
using ClimateMachine.Atmos:
    AtmosPhysics,
    AtmosModel,
    KennedyGruberSplitForm,
    DryModel,
    NoPrecipitation,
    NoRadiation,
    NTracers,
    vars_state,
    Gravity,
    HydrostaticState,
    AtmosAcousticGravityLinearModel,
    parameter_set
using ClimateMachine.TurbulenceClosures
using ClimateMachine.Orientations:
    SphericalOrientation, gravitational_potential, altitude, latitude, longitude
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = false

const ntracers = 1

function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 10
    numelem_vert = 5

    timeend = 33 * 60 * 60
    # timeend = 33 * 60 * 60 # Full simulation
    outputtime = 60 * 60

    expected_result = Dict()
    expected_result[Float32] = 9.5066030866432000e+13
    expected_result[Float64] = 9.5073452847149594e+13

    for FT in (Float64, Float32)
        for split_explicit_implicit in (false, true)
            result = test_run(
                mpicomm,
                polynomialorder,
                numelem_horz,
                numelem_vert,
                timeend,
                outputtime,
                ArrayType,
                FT,
                split_explicit_implicit,
            )
            @test result ≈ expected_result[FT]
        end
    end
end

function test_run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
    split_explicit_implicit,
)

    setup = AcousticWaveSetup{FT}()

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + setup.domain_height),
        nelem = numelem_vert,
    )
    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = equiangular_cubed_sphere_warp,
    )

    T_profile = IsothermalProfile(param_set, setup.T_ref)
    δ_χ = @SVector [FT(ii) for ii in 1:ntracers]

    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = HydrostaticState(T_profile; subtract_off = false),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        #tracers = NTracers{length(δ_χ), FT}(δ_χ),
    )
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        physics;
        init_state_prognostic = setup,
        source = (),
        equations_form = KennedyGruberSplitForm(),
    )
    linearmodel = AtmosAcousticGravityLinearModel(model)

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    lineardg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = dg.state_auxiliary,
    )

    # determine the time step
    element_size = (setup.domain_height / numelem_vert)
    acoustic_speed = soundspeed_air(param_set, FT(setup.T_ref))
    dt_factor = 50
    dt = dt_factor * element_size / acoustic_speed / polynomialorder^2
    # Adjust the time step so we exactly hit 1 hour for VTK output
    dt = 60 * 60 / ceil(60 * 60 / dt)
    dt = 3 * min_node_distance(grid) / 330
    nsteps = ceil(Int, timeend / dt)

    Q = init_ode_state(dg, FT(0))

    linearsolver = ManyColumnLU()
    # linearsolver = BatchedGeneralizedMinimalResidual(
    #    lineardg,
    #    Q;
    #    atol = 1.0e-6, #sqrt(eps(FT)) * 0.01,
    #    rtol = 1.0e-8, #sqrt(eps(FT)) * 0.01,
    #    # Maximum number of Krylov iterations in a column
    #)


    if split_explicit_implicit
        rem_dg = remainder_DGModel(
            dg,
            (lineardg,);
            numerical_flux_first_order = (
                dg.numerical_flux_first_order,
                (lineardg.numerical_flux_first_order,),
            ),
        )
    end
    odesolver = ARK2GiraldoKellyConstantinescu(
        split_explicit_implicit ? rem_dg : dg,
        lineardg,
        LinearBackwardEulerSolver(
            linearsolver;
            isadjustable = true,
            preconditioner_update_freq = -1,
        ),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = split_explicit_implicit,
    )
    @test getsteps(odesolver) == 0

    filterorder = 18
    filter = ExponentialFilter(grid, 0, filterorder)
    cbfilter = EveryXSimulationSteps(1) do
        Filters.apply!(Q, :, grid, filter, direction = VerticalDirection())
        nothing
    end

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end
    callbacks = (cbinfo,)# cbfilter)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_acousticwave" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_dt$(dt_factor)x_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(
        Q,
        odesolver;
        numberofsteps = nsteps,
        adjustfinalstep = false,
        callbacks = callbacks,
    )

    @test getsteps(odesolver) == nsteps

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end

Base.@kwdef struct AcousticWaveSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function (setup::AcousticWaveSetup)(problem, bl, state, aux, localgeo, t)
    # callable to set initial conditions
    FT = eltype(state)
    param_set = parameter_set(bl)

    λ = longitude(bl, aux)
    φ = latitude(bl, aux)
    z = altitude(bl, aux)

    β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(FT(π) * β)) / 2
    g = sin(setup.nv * FT(π) * z / setup.domain_height)
    Δp = setup.γ * f * g
    p = aux.ref_state.p + Δp

    ts = PhaseDry_pT(param_set, p, setup.T_ref)
    q_pt = PhasePartition(ts)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(ts)

    state.ρ = air_density(ts)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.energy.ρe = state.ρ * (e_int + e_pot)

    #state.tracers.ρχ = @SVector [FT(ii) for ii in 1:ntracers]
    nothing
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "acousticwave",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
