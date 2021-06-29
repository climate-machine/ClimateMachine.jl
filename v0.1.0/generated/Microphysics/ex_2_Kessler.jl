include("KinematicModel.jl")

function vars_state_conservative(m::KinematicModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
        ρq_liq::FT
        ρq_ice::FT
        ρq_rai::FT
    end
end

function vars_state_auxiliary(m::KinematicModel, FT)
    @vars begin

        p::FT
        z::FT

        u::FT
        w::FT
        q_tot::FT
        q_vap::FT
        q_liq::FT
        q_ice::FT
        q_rai::FT
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
        T::FT
        S::FT
        RH::FT
        rain_w::FT

        #src_cloud_liq::FT
        #src_cloud_ice::FT
        #src_acnv::FT
        #src_accr::FT
        #src_rain_evap::FT
        #flag_rain::FT
        #flag_cloud_liq::FT
        #flag_cloud_ice::FT
    end
end

function init_kinematic_eddy!(eddy_model, state, aux, (x, y, z), t)
    FT = eltype(state)
    _grav::FT = grav(param_set)

    dc = eddy_model.data_config

    q_pt_0 = PhasePartition(dc.qt_0)
    R_m, cp_m, cv_m, γ = gas_constants(param_set, q_pt_0)
    T::FT = dc.θ_0 * (aux.p / dc.p_1000)^(R_m / cp_m)
    ρ::FT = aux.p / R_m / T
    state.ρ = ρ

    state.ρq_tot = ρ * dc.qt_0
    state.ρq_liq = ρ * q_pt_0.liq
    state.ρq_ice = ρ * q_pt_0.ice
    state.ρq_rai = ρ * FT(0)

    ρu::FT =
        dc.wmax * dc.xmax / dc.zmax *
        cos(π * z / dc.zmax) *
        cos(2 * π * x / dc.xmax)
    ρw::FT = 2 * dc.wmax * sin(π * z / dc.zmax) * sin(2 * π * x / dc.xmax)
    state.ρu = SVector(ρu, FT(0), ρw)
    u::FT = ρu / ρ
    w::FT = ρw / ρ

    e_kin::FT = 1 // 2 * (u^2 + w^2)
    e_pot::FT = _grav * z
    e_int::FT = internal_energy(param_set, T, q_pt_0)
    e_tot::FT = e_kin + e_pot + e_int
    state.ρe = ρ * e_tot

    return nothing
end

function kinematic_model_nodal_update_auxiliary_state!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _grav::FT = grav(param_set)

    aux.u = state.ρu[1] / state.ρ
    aux.w = state.ρu[3] / state.ρ

    aux.q_tot = state.ρq_tot / state.ρ
    aux.q_liq = state.ρq_liq / state.ρ
    aux.q_ice = state.ρq_ice / state.ρ
    aux.q_rai = state.ρq_rai / state.ρ
    aux.q_vap = aux.q_tot - aux.q_liq - aux.q_ice

    aux.e_tot = state.ρe / state.ρ
    aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
    aux.e_pot = _grav * aux.z
    aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot

    q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
    aux.T = air_temperature(param_set, aux.e_int, q)
    ts_neq = TemperatureSHumNonEquil(param_set, aux.T, state.ρ, q)
    aux.S = max(0, aux.q_vap / q_vap_saturation(ts_neq) - FT(1)) * FT(100)
    aux.RH = aux.q_vap / q_vap_saturation(ts_neq) * FT(100)

    aux.rain_w = terminal_velocity(param_set, aux.q_rai, state.ρ)

    #ts_eq = TemperatureSHumEquil(param_set, aux.T, state.ρ, aux.q_tot)
    #q_eq = PhasePartition(ts_eq)
    #aux.src_cloud_liq = conv_q_vap_to_q_liq(q_eq, q)
    #aux.src_cloud_ice = conv_q_vap_to_q_ice(q_eq, q)
    #aux.src_acnv = conv_q_liq_to_q_rai_acnv(aux.q_liq)
    #aux.src_accr = conv_q_liq_to_q_rai_accr(aux.q_liq, aux.q_rai, state.ρ)
    #aux.src_rain_evap = conv_q_rai_to_q_vap(aux.q_rai, q, aux.T, aux.p, state.ρ)
    #aux.flag_cloud_liq = FT(0)
    #aux.flag_cloud_ice = FT(0)
    #aux.flag_rain = FT(0)
    #if (aux.q_liq >= FT(0))

    #end
    #if (aux.q_ice >= FT(0))

    #end
    #if (aux.q_rai >= FT(0))

    #end
end

function boundary_state!(
    ::RusanovNumericalFlux,
    m::KinematicModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
)
    #state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
    state⁺.ρq_rai = -state⁻.ρq_rai
end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    u = state.ρu / state.ρ
    rain_w = terminal_velocity(param_set, state.ρq_rai / state.ρ, state.ρ)
    nu = nM[1] * u[1] + nM[3] * max(u[3], rain_w, u[3] - rain_w)

    return abs(nu)
end

@inline function flux_first_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    rain_w = terminal_velocity(param_set, state.ρq_rai / state.ρ, state.ρ)

    flux.ρq_tot = SVector(
        state.ρu[1] * state.ρq_tot / state.ρ,
        FT(0),
        state.ρu[3] * state.ρq_tot / state.ρ,
    )
    flux.ρq_liq = SVector(
        state.ρu[1] * state.ρq_liq / state.ρ,
        FT(0),
        state.ρu[3] * state.ρq_liq / state.ρ,
    )
    flux.ρq_ice = SVector(
        state.ρu[1] * state.ρq_ice / state.ρ,
        FT(0),
        state.ρu[3] * state.ρq_ice / state.ρ,
    )
    flux.ρq_rai = SVector(
        state.ρu[1] * state.ρq_rai / state.ρ,
        FT(0),
        (state.ρu[3] / state.ρ - rain_w) * state.ρq_rai,
    )

    flux.ρe = SVector(
        state.ρu[1] / state.ρ * (state.ρe + aux.p),
        FT(0),
        state.ρu[3] / state.ρ * (state.ρe + aux.p),
    )

end

function source!(
    m::KinematicModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)

    FT = eltype(state)
    _grav::FT = grav(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _cv_v::FT = cv_v(param_set)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    e_tot = state.ρe / state.ρ
    q_tot = state.ρq_tot / state.ρ
    q_liq = state.ρq_liq / state.ρ
    q_ice = state.ρq_ice / state.ρ
    q_rai = state.ρq_rai / state.ρ
    u = state.ρu[1] / state.ρ
    w = state.ρu[3] / state.ρ
    e_int = e_tot - 1 // 2 * (u^2 + w^2) - _grav * aux.z

    q = PhasePartition(q_tot, q_liq, q_ice)
    T = air_temperature(param_set, e_int, q)

    ts_eq = TemperatureSHumEquil(param_set, T, state.ρ, q_tot)
    q_eq = PhasePartition(ts_eq)

    source.ρq_tot = FT(0)
    source.ρq_liq = FT(0)
    source.ρq_ice = FT(0)
    source.ρq_rai = FT(0)
    source.ρe = FT(0)

    source.ρq_liq += state.ρ * conv_q_vap_to_q_liq(param_set, q_eq, q)
    source.ρq_ice += state.ρ * conv_q_vap_to_q_ice(param_set, q_eq, q)

    src_q_rai_acnv = conv_q_liq_to_q_rai_acnv(param_set, q_liq)
    src_q_rai_accr = conv_q_liq_to_q_rai_accr(param_set, q_liq, q_rai, state.ρ)
    src_q_rai_evap = conv_q_rai_to_q_vap(param_set, q_rai, q, T, aux.p, state.ρ)

    src_q_rai_tot = src_q_rai_acnv + src_q_rai_accr + src_q_rai_evap

    source.ρq_liq -= state.ρ * (src_q_rai_acnv + src_q_rai_accr)
    source.ρq_rai += state.ρ * src_q_rai_tot
    source.ρq_tot -= state.ρ * src_q_rai_tot
    source.ρe -=
        state.ρ * src_q_rai_tot * (_e_int_v0 - (_cv_v - _cv_d) * (T - _T_0))
end

function main()

    FT = Float64

    N = 4

    Δx = FT(20)
    Δy = FT(1)
    Δz = FT(20)
    resolution = (Δx, Δy, Δz)

    xmax = 1500
    ymax = 10
    zmax = 1500

    wmax = FT(0.6)  # max velocity of the eddy  [m/s]
    θ_0 = FT(289) # init. theta value (const) [K]
    p_0 = FT(101500) # surface pressure [Pa]
    p_1000 = FT(100000) # reference pressure in theta definition [Pa]
    qt_0 = FT(7.5 * 1e-3) # init. total water specific humidity (const) [kg/kg]
    z_0 = FT(0) # surface height

    t_ini = FT(0)
    t_end = FT(30 * 60)
    dt = FT(5)
    #CFL = FT(1.75)
    filter_freq = 1
    output_freq = 72

    driver_config = config_kinematic_eddy(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        wmax,
        θ_0,
        p_0,
        p_1000,
        qt_0,
        z_0,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t_ini,
        t_end,
        driver_config;
        ode_dt = dt,
        init_on_cpu = true,
        #Courant_number = CFL,
    )

    model = driver_config.bl

    mpicomm = MPI.COMM_WORLD

    ρq_liq_ind = varsindex(vars_state_conservative(model, FT), :ρq_liq)
    ρq_ice_ind = varsindex(vars_state_conservative(model, FT), :ρq_ice)
    ρq_rai_ind = varsindex(vars_state_conservative(model, FT), :ρq_rai)

    q_tot_ind = varsindex(vars_state_auxiliary(model, FT), :q_tot)
    q_vap_ind = varsindex(vars_state_auxiliary(model, FT), :q_vap)
    q_liq_ind = varsindex(vars_state_auxiliary(model, FT), :q_liq)
    q_ice_ind = varsindex(vars_state_auxiliary(model, FT), :q_ice)
    q_rai_ind = varsindex(vars_state_auxiliary(model, FT), :q_rai)
    S_ind = varsindex(vars_state_auxiliary(model, FT), :S)
    rain_w_ind = varsindex(vars_state_auxiliary(model, FT), :rain_w)

    cb_tmar_filter =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (ρq_liq_ind[1], ρq_ice_ind[1], ρq_rai_ind[1]),
                solver_config.dg.grid,
                TMARFilter(),
            )
            nothing
        end

    vtkdir = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk"))
    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    step = [0]
    cb_vtk =
        GenericCallbacks.EveryXSimulationSteps(output_freq) do (init = false)
            out_dirname = @sprintf(
                "new_ex_2_mpirank%04d_step%04d",
                MPI.Comm_rank(mpicomm),
                step[1]
            )
            out_path_prefix = joinpath(vtkdir, out_dirname)
            @info "doing VTK output" out_path_prefix
            writevtk(
                out_path_prefix,
                solver_config.Q,
                solver_config.dg,
                flattenednames(vars_state_conservative(model, FT)),
                solver_config.dg.state_auxiliary,
                flattenednames(vars_state_auxiliary(model, FT)),
            )
            step[1] += 1
            nothing
        end

    result = ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cb_tmar_filter, cb_vtk),
        check_euclidean_distance = true,
    )

    max_S = maximum(abs.(solver_config.dg.state_auxiliary[:, S_ind, :]))
    @test max_S < FT(0.25)
    @test max_S > FT(0)

    max_q_tot = maximum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    @test max_q_tot < FT(0.0077)

    max_q_ice = maximum(abs.(solver_config.dg.state_auxiliary[:, q_ice_ind, :]))
    @test isequal(max_q_ice, FT(0))

    max_q_liq = max(solver_config.dg.state_auxiliary[:, q_liq_ind, :]...)
    min_q_liq = min(solver_config.dg.state_auxiliary[:, q_liq_ind, :]...)
    @test max_q_liq < FT(1e-3)
    @test abs(min_q_liq) < FT(1e-5)

    max_q_rai = max(solver_config.dg.state_auxiliary[:, q_rai_ind, :]...)
    min_q_rai = min(solver_config.dg.state_auxiliary[:, q_rai_ind, :]...)
    @test max_q_rai < FT(3e-5)
    @test abs(min_q_rai) < FT(3e-8)

    max_rain_w = max(solver_config.dg.state_auxiliary[:, rain_w_ind, :]...)
    min_rain_w = min(solver_config.dg.state_auxiliary[:, rain_w_ind, :]...)
    @test max_rain_w < FT(4)
    @test isequal(min_rain_w, FT(0))
end

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

