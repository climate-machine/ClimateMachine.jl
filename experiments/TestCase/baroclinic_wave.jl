#!/usr/bin/env julia --project

using ArgParse
using LinearAlgebra
using StaticArrays
using Test

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics:
    air_density, air_temperature, total_energy, internal_energy, PhasePartition
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.Spectra: compute_gaussian!

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, Omega, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_baroclinic_wave!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    # parameters
    param_set = parameter_set(bl)
    _grav::FT = grav(param_set)
    _R_d::FT = R_d(param_set)
    _Ω::FT = Omega(param_set)
    _a::FT = planet_radius(param_set)
    _p_0::FT = MSLP(param_set)

    k::FT = 3
    T_E::FT = 310
    T_P::FT = 240
    T_0::FT = 0.5 * (T_E + T_P)
    Γ::FT = 0.005
    A::FT = 1 / Γ
    B::FT = (T_0 - T_P) / T_0 / T_P
    C::FT = 0.5 * (k + 2) * (T_E - T_P) / T_E / T_P
    b::FT = 2
    H::FT = _R_d * T_0 / _grav
    z_t::FT = 15e3
    λ_c::FT = π / 9
    φ_c::FT = 2 * π / 9
    d_0::FT = _a / 6
    V_p::FT = 1
    M_v::FT = 0.608
    p_w::FT = 34e3                 # Pressure width parameter for specific humidity
    η_crit::FT = p_w / _p_0        # Critical pressure coordinate
    q_0::FT = 0.018                # Maximum specific humidity (default: 0.018)
    q_t::FT = 1e-12                # Specific humidity above artificial tropopause
    φ_w::FT = 2π / 9               # Specific humidity latitude wind parameter

    # grid
    φ = latitude(bl.orientation, aux)
    λ = longitude(bl.orientation, aux)
    z = altitude(bl.orientation, param_set, aux)
    r::FT = z + _a
    γ::FT = 1 # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

    # convenience functions for temperature and pressure
    τ_z_1::FT = exp(Γ * z / T_0)
    τ_z_2::FT = 1 - 2 * (z / b / H)^2
    τ_z_3::FT = exp(-(z / b / H)^2)
    τ_1::FT = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2::FT = C * τ_z_2 * τ_z_3
    τ_int_1::FT = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2::FT = C * z * τ_z_3
    I_T::FT =
        (cos(φ) * (1 + γ * z / _a))^k -
        k / (k + 2) * (cos(φ) * (1 + γ * z / _a))^(k + 2)

    # base state virtual temperature, pressure, specific humidity, density
    T_v::FT = (τ_1 - τ_2 * I_T)^(-1)
    p::FT = _p_0 * exp(-_grav / _R_d * (τ_int_1 - τ_int_2 * I_T))

    # base state velocity
    U::FT =
        _grav * k / _a *
        τ_int_2 *
        T_v *
        (
            (cos(φ) * (1 + γ * z / _a))^(k - 1) -
            (cos(φ) * (1 + γ * z / _a))^(k + 1)
        )
    u_ref::FT =
        -_Ω * (_a + γ * z) * cos(φ) +
        sqrt((_Ω * (_a + γ * z) * cos(φ))^2 + (_a + γ * z) * cos(φ) * U)
    v_ref::FT = 0
    w_ref::FT = 0

    # velocity perturbations
    F_z::FT = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
    if z > z_t
        F_z = FT(0)
    end
    d::FT = _a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
    c3::FT = cos(π * d / 2 / d_0)^3
    s1::FT = sin(π * d / 2 / d_0)
    if 0 < d < d_0 && d != FT(_a * π)
        u′::FT =
            -16 * V_p / 3 / sqrt(3) *
            F_z *
            c3 *
            s1 *
            (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) /
            sin(d / _a)
        v′::FT =
            16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
            sin(d / _a)
    else
        u′ = FT(0)
        v′ = FT(0)
    end
    w′::FT = 0
    u_sphere = SVector{3, FT}(u_ref + u′, v_ref + v′, w_ref + w′)
    u_cart = sphr_to_cart_vec(bl.orientation, u_sphere, aux)

    if moisture_model(bl) isa DryModel
        q_tot = FT(0)
    else
        ## Compute moisture profile
        ## Pressure coordinate η
        ## η_crit = p_t / p_w ; p_t = 10000 hPa, p_w = 340 hPa
        η = p / _p_0
        if η > η_crit
            q_tot = q_0 * exp(-(φ / φ_w)^4) * exp(-((η - 1) * _p_0 / p_w)^2)
        else
            q_tot = q_t
        end
    end
    phase_partition = PhasePartition(q_tot)

    ## temperature & density
    T::FT = T_v / (1 + M_v * q_tot)
    ρ::FT = air_density(param_set, T, p, phase_partition)

    ## potential & kinetic energy
    e_pot::FT = gravitational_potential(bl.orientation, aux)
    e_kin::FT = 0.5 * u_cart' * u_cart
    e_tot::FT = total_energy(param_set, e_kin, e_pot, T, phase_partition)

    ## Assign state variables
    state.ρ = ρ
    state.ρu = ρ * u_cart
    state.energy.ρe = ρ * e_tot

    if !(moisture_model(bl) isa DryModel)
        state.moisture.ρq_tot = ρ * q_tot
    end

    nothing
end

function config_baroclinic_wave(FT, poly_order, resolution, with_moisture)
    # Set up a reference state for linearization of equations
    temp_profile_ref =
        DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))
    ref_state = HydrostaticState(temp_profile_ref)

    # Set up the atmosphere model
    exp_name = "BaroclinicWave"
    domain_height::FT = 30e3 # distance between surface and top of atmosphere (m)
    if with_moisture
        hyperdiffusion = EquilMoistBiharmonic(FT(8 * 3600))
        moisture = EquilMoist()
        source = (Gravity(), Coriolis(), RemovePrecipitation(true)) # precipitation is default to NoPrecipitation() as 0M microphysics
    else
        hyperdiffusion = DryBiharmonic(FT(8 * 3600))
        moisture = DryModel()
        source = (Gravity(), Coriolis())
    end
    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = ConstantKinematicViscosity(FT(0)),
        hyperdiffusion = hyperdiffusion,
        moisture = moisture,
    )
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        physics;
        init_state_prognostic = init_baroclinic_wave!,
        source = source,
        equations_form = KennedyGruberSplitForm(),
    )

    config = ClimateMachine.AtmosGCMConfiguration(
        exp_name,
        poly_order,
        resolution,
        domain_height,
        param_set,
        init_baroclinic_wave!;
        model = model,
    )

    return config
end

function main()
    # add a command line argument to specify whether to use a moist setup
    # TODO: this will move to the future namelist functionality
    bw_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(bw_args, "BaroclinicWave")
    @add_arg_table! bw_args begin
        "--with-moisture"
        help = "use a moist setup"
        action = :store_const
        constant = true
        default = false
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = bw_args)
    with_moisture = cl_args["with_moisture"]

    # Driver configuration parameters
    FT = Float64                             # floating type precision
    poly_order = (3, 3)                      # discontinuous Galerkin polynomial order
    n_horz = 10                               # horizontal element number
    n_vert = 5                               # vertical element number
    n_days::FT = 100
    timestart::FT = 0                        # start time (s)
    timeend::FT = n_days * day(param_set)    # end time (s)

    # Set up driver configuration
    driver_config =
        config_baroclinic_wave(FT, poly_order, (n_horz, n_vert), with_moisture)

    # Set up experiment
    ode_solver_type = ClimateMachine.IMEXSolverType(
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
        split_explicit_implicit = false,
        discrete_splitting = true,
    )

    CFL = FT(3.0) # target acoustic CFL number

    # time step is computed such that the horizontal acoustic Courant number is CFL
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        Courant_number = CFL,
        ode_solver_type = ode_solver_type,
        CFL_direction = VerticalDirection(),
        diffdir = HorizontalDirection(),
    )

    # Set up diagnostics
    dgn_config = config_diagnostics(FT, driver_config)

    # Set up user-defined callbacks
    filterorder = 20
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            state_auxiliary = solver_config.dg.state_auxiliary,
        )
        nothing
    end

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot",),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    # Run the model
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        #user_callbacks = (cbfilter,),
        #user_callbacks = (cbtmarfilter, cbfilter),
        check_euclidean_distance = true,
    )
end

function config_diagnostics(FT, driver_config)
    interval = "12shours" # chosen to allow diagnostics every 12 simulated hours

    _planet_radius = FT(planet_radius(param_set))

    info = driver_config.config_info

    # Setup diagnostic grid(s)
    nlats = 128

    sinθ, wts = compute_gaussian!(nlats)
    lats = asin.(sinθ) .* 180 / π
    lons = 180.0 ./ nlats * collect(FT, 1:1:(2nlats))[:] .- 180.0

    boundaries = [
        FT(lats[1]) FT(lons[1]) _planet_radius
        FT(lats[end]) FT(lons[end]) FT(_planet_radius + info.domain_height)
    ]

    lvls = collect(range(
        boundaries[1, 3],
        boundaries[2, 3],
        step = FT(1000), # in m
    ))

    #boundaries = [
    #    FT(-90.0) FT(-180.0) _planet_radius
    #    FT(90.0) FT(180.0) FT(_planet_radius + info.domain_height)
    #]

    #resolution = (FT(2), FT(2), FT(1000)) # in (deg, deg, m)

    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries;
        axes = [lats, lons, lvls],
    )

    dgngrp = setup_atmos_default_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    ds_dgngrp = setup_atmos_spectra_diagnostics(
        AtmosGCMConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )

    return ClimateMachine.DiagnosticsConfiguration([dgngrp, ds_dgngrp])
end

main()
