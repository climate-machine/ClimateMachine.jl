using Dates
using FileIO
using KernelAbstractions
using MPI
using NCDatasets
using Printf
using Random
using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init(diagnostics = "1steps")
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DiagnosticsMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.MPIStateArrays
using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# Need to import these to define new diagnostic variables and groups.
import ..DiagnosticsMachine:
    Settings,
    dv_name,
    dv_attrib,
    dv_args,
    dv_project,
    dv_scale,
    dv_PointwiseDiagnostic,
    dv_HorizontalAverage

# Define some new diagnostic variables.
@horizontal_average(
    AtmosLESConfigType,
    yvel,
) do (atmos::AtmosModel, states::States, curr_time, cache)
    states.prognostic.ρu[2] / states.prognostic.ρ
end

@pointwise_diagnostic(
    AtmosLESConfigType,
    zvel,
) do (atmos::AtmosModel, states::States, curr_time, cache)
    states.prognostic.ρu[3] / states.prognostic.ρ
end

# Define a new diagnostics group with some pre-defined diagnostic
# variables as well as the new ones above.
@diagnostics_group(
    "DMTest",
    AtmosLESConfigType,
    Nothing,
    (_...) -> nothing,
    NoInterpolation,
    u,
    v,
    w,
    rho,
    yvel,
    zvel,
)

# Make sure DiagnosticsMachine did what it was supposed to do.
@testset "DiagnosticsMachine interface" begin
    @test_throws UndefVarError ALESCT_HA_foo()
    yv = ALESCT_HA_yvel()
    @test yv isa HorizontalAverage
    yvname = dv_name(AtmosLESConfigType(), yv)
    @test yvname == "yvel"
    @test yvname ∈
          keys(DiagnosticsMachine.AllDiagnosticVars[AtmosLESConfigType])

    @test_throws UndefVarError ALESCT_PD_bar()
    zv = ALESCT_PD_zvel()
    @test zv isa PointwiseDiagnostic
    zvname = dv_name(AtmosLESConfigType(), zv)
    @test zvname == "zvel"
    @test zvname ∈
          keys(DiagnosticsMachine.AllDiagnosticVars[AtmosLESConfigType])
end

# Set up a simple experiment to run the diagnostics group.

include("sin_init.jl")

function main()
    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(25)
    Δv = FT(25)
    resolution = (Δh, Δh, Δv)

    xmax = FT(500)
    ymax = FT(500)
    zmax = FT(500)

    t0 = FT(0)
    dt = FT(0.01)
    timeend = dt

    driver_config = ClimateMachine.AtmosLESConfiguration(
        "DMTest",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_sin_test!,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
        init_on_cpu = true,
    )
    dm_dgngrp = DMTest("1steps", driver_config.name)
    dgn_config = ClimateMachine.DiagnosticsConfiguration([dm_dgngrp])

    ClimateMachine.invoke!(solver_config, diagnostics_config = dgn_config)

    # Check the output from the diagnostics group.
    @testset "DiagnosticsMachine correctness" begin
        mpicomm = solver_config.mpicomm
        mpirank = MPI.Comm_rank(mpicomm)
        if mpirank == 0
            nm = driver_config.name * "_DMTest.nc"
            ds = Dataset(joinpath(ClimateMachine.Settings.output_dir, nm), "r")
            ds_u = ds["u"][:]
            ds_yvel = ds["yvel"][:]
            ds_zvel = ds["zvel"][:]
            close(ds)
        end
        Q =
            array_device(solver_config.Q) isa CPU ? solver_config.Q :
            Array(solver_config.Q)
        havg_rho = compute_havg(solver_config, view(Q, :, 1:1, :))
        havg_u = compute_havg(solver_config, view(Q, :, 2:2, :))
        v = view(Q, :, 3:3, :) ./ view(Q, :, 1:1, :)
        havg_v = compute_havg(solver_config, v)
        if mpirank == 0
            havg_u ./= havg_rho
            @test all(ds_u[:, 2] .≈ havg_u)
            @test all(ds_yvel[:, 2] .≈ havg_v)

            realelems = solver_config.dg.grid.topology.realelems
            w = view(Q, :, 4, realelems) ./ view(Q, :, 1, realelems)
            @test all(ds_zvel[:, :, 2] .≈ w)
        end
    end

    nothing
end

function compute_havg(solver_config, field)
    mpicomm = solver_config.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)
    grid = solver_config.dg.grid
    grid_info = basic_grid_info(grid)
    topl_info = basic_topology_info(grid.topology)
    Nqh = grid_info.Nqh
    Nqk = grid_info.Nqk
    nvertelem = topl_info.nvertelem
    nhorzrealelem = topl_info.nhorzrealelem

    function arrange_array(A, dim = :)
        A = array_device(A) isa CPU ? A : Array(A)
        reshape(
            view(A, :, dim, grid.topology.realelems),
            Nqh,
            Nqk,
            nvertelem,
            nhorzrealelem,
        )
    end

    field = arrange_array(field)
    MH = arrange_array(grid.vgeo, grid.MHid)
    full_field = MPI.Reduce!(sum(field .* MH, dims = (1, 4))[:], +, 0, mpicomm)
    full_MH = MPI.Reduce!(sum(MH, dims = (1, 4))[:], +, 0, mpicomm)
    if mpirank == 0
        return full_field ./ full_MH
    else
        return nothing
    end
end

main()
