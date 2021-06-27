#!/usr/bin/env julia --project
using Test
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorders
using ClimateMachine.Ocean.HydrostaticBoussinesq
using ClimateMachine.Ocean.ShallowWater
using ClimateMachine.Ocean.SplitExplicit: VerticalIntegralModel
using ClimateMachine.Ocean.OceanProblems

using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.BalanceLaws
using ClimateMachine.BalanceLaws: number_state_auxiliary
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.VTK

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function test_vertical_integral_model(::Type{FT}, time; refDat = ()) where {FT}
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()


    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D,
        periodicity = (true, true),
        boundary = ((0, 0), (0, 0)),
    )
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
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    problem = SimpleBox{FT}(Lˣ, Lʸ, H)

    model_3D = HydrostaticBoussinesqModel{FT}(
        param_set,
        problem;
        cʰ = FT(1),
        αᵀ = FT(0),
        κʰ = FT(0),
        κᶻ = FT(0),
        fₒ = FT(0),
        β = FT(0),
    )

    model_2D = ShallowWaterModel{FT}(
        param_set,
        problem,
        ShallowWater.ConstantViscosity{FT}(model_3D.νʰ),
        nothing;
        c = FT(1),
        fₒ = FT(0),
        β = FT(0),
    )

    integral_bl = VerticalIntegralModel(model_3D)

    integral_model = DGModel(
        integral_bl,
        grid_3D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    dg_3D = DGModel(
        model_3D,
        grid_3D,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q_3D = init_ode_state(dg_3D, FT(time); init_on_cpu = true)
    Q_2D = init_ode_state(dg_2D, FT(time); init_on_cpu = true)
    Q_int = integral_model.state_auxiliary

    state_check = ClimateMachine.StateCheck.sccreate(
        [
            (Q_int, "∫u")
            (Q_2D, "U")
        ],
        1;
        prec = 12,
    )

    update_auxiliary_state!(integral_model, integral_bl, Q_3D, time)
    GenericCallbacks.call!(state_check, nothing, nothing, nothing, nothing)

    ClimateMachine.StateCheck.scprintref(state_check)
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(state_check, refDat)
    end

    return nothing
end

#################
# RUN THE TESTS #
#################
const FT = Float64

const N = 4
const Nˣ = 5
const Nʸ = 5
const Nᶻ = 8
const Lˣ = 1e6  # m
const Lʸ = 1e6  # m
const H = 400  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)

const cʰ = 1  # typical of ocean internal-wave speed
const cᶻ = 0

@testset "$(@__FILE__)" begin
    include("../refvals/test_vertical_integral_model_refvals.jl")

    times =
        [0, 86400, 30 * 86400, 365 * 86400, 10 * 365 * 86400, 100 * 365 * 86400]
    for (index, time) in enumerate(times)
        @testset "$(time)" begin
            test_vertical_integral_model(FT, time, refDat = refVals[index])
        end
    end
end
