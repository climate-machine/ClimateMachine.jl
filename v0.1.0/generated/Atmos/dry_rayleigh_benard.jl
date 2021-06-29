using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using Printf

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.MoistThermodynamics:
    TemperatureSHumEquil_given_pressure, internal_energy
using ClimateMachine.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

struct DryRayleighBenardConvectionDataConfig{FT}
    xmin::FT
    ymin::FT
    zmin::FT
    xmax::FT
    ymax::FT
    zmax::FT
    T_bot::FT
    T_lapse::FT
    T_top::FT
end

function init_problem!(bl, state, aux, (x, y, z), t)
    dc = bl.data_config
    FT = eltype(state)

    _R_d::FT = R_d(bl.param_set)
    _cp_d::FT = cp_d(bl.param_set)
    _grav::FT = grav(bl.param_set)
    _cv_d::FT = cv_d(bl.param_set)
    _MSLP::FT = MSLP(bl.param_set)

    γ::FT = _cp_d / _cv_d
    δT =
        sinpi(6 * z / (dc.zmax - dc.zmin)) *
        cospi(6 * z / (dc.zmax - dc.zmin)) + rand()
    δw =
        sinpi(6 * z / (dc.zmax - dc.zmin)) *
        cospi(6 * z / (dc.zmax - dc.zmin)) + rand()
    ΔT = _grav / _cv_d * z + δT
    T = dc.T_bot - ΔT
    P = _MSLP * (T / dc.T_bot)^(_grav / _R_d / dc.T_lapse)
    ρ = P / (_R_d * T)

    q_tot = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ts = TemperatureSHumEquil_given_pressure(bl.param_set, T, P, q_tot)

    ρu, ρv, ρw = FT(0), FT(0), ρ * δw

    e_int = internal_energy(ts)
    e_kin = FT(1 / 2) * δw^2

    ρe_tot = ρ * (e_int + e_pot + e_kin)
    state.ρ = ρ
    state.ρu = SVector(ρu, ρv, ρw)
    state.ρe = ρe_tot
    state.moisture.ρq_tot = FT(0)
    ρχ = zero(FT)
    if z <= 100
        ρχ += FT(0.1) * (cospi(z / 2 / 100))^2
    end
    state.tracers.ρχ = SVector{1, FT}(ρχ)
end

function config_problem(FT, N, resolution, xmax, ymax, zmax)

    T_bot = FT(299)

    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)

    T_lapse = FT(_grav / _cp_d)
    T_top = T_bot - T_lapse * zmax

    ntracers = 1
    δ_χ = SVector{ntracers, FT}(1)

    C_smag = FT(0.23)
    data_config = DryRayleighBenardConvectionDataConfig{FT}(
        0,
        0,
        0,
        xmax,
        ymax,
        zmax,
        T_bot,
        T_lapse,
        FT(T_bot - T_lapse * zmax),
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = Vreman(C_smag),
        source = (Gravity(),),
        boundarycondition = (
            AtmosBC(
                momentum = Impenetrable(NoSlip()),
                energy = PrescribedTemperature((state, aux, t) -> T_bot),
            ),
            AtmosBC(
                momentum = Impenetrable(NoSlip()),
                energy = PrescribedTemperature((state, aux, t) -> T_top),
            ),
        ),
        tracers = NTracers{ntracers, FT}(δ_χ),
        init_state_conservative = init_problem!,
        data_config = data_config,
    )

    ode_solver = ClimateMachine.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "DryRayleighBenardConvection",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_problem!,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(interval, driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    FT = Float64

    N = 4

    Δh = FT(10)

    t0 = FT(0)

    CFLmax = FT(5)
    timeend = FT(1000)
    xmax, ymax, zmax = FT(250), FT(250), FT(500)

    @testset "DryRayleighBenardTest" begin
        for Δh in Δh
            Δv = Δh
            resolution = (Δh, Δh, Δv)
            driver_config = config_problem(FT, N, resolution, xmax, ymax, zmax)
            solver_config = ClimateMachine.SolverConfiguration(
                t0,
                timeend,
                driver_config,
                init_on_cpu = true,
                Courant_number = CFLmax,
            )
            dgn_config = config_diagnostics(driver_config)

            cbtmarfilter =
                GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
                    Filters.apply!(
                        solver_config.Q,
                        6,
                        solver_config.dg.grid,
                        TMARFilter(),
                    )
                    nothing
                end
            result = ClimateMachine.invoke!(
                solver_config;
                diagnostics_config = dgn_config,
                user_callbacks = (cbtmarfilter,),
                check_euclidean_distance = true,
            )

            @test isapprox(result, FT(1); atol = 1.5e-2)
        end
    end
end

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

