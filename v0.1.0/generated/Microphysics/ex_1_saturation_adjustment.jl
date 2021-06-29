include("KinematicModel.jl")

function vars_state_conservative(m::KinematicModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
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
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
        T::FT
        S::FT
        RH::FT
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

    aux.e_tot = state.ρe / state.ρ
    aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
    aux.e_pot = _grav * aux.z
    aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot

    ts = PhaseEquil(param_set, aux.e_int, state.ρ, aux.q_tot)
    pp = PhasePartition(ts)

    aux.T = ts.T
    aux.q_vap = aux.q_tot - pp.liq - pp.ice
    aux.q_liq = pp.liq
    aux.q_ice = pp.ice

    q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
    ts_neq = TemperatureSHumNonEquil(param_set, aux.T, state.ρ, q)
    aux.S = max(0, aux.q_vap / q_vap_saturation(ts_neq) - FT(1)) * FT(100)
    aux.RH = aux.q_vap / q_vap_saturation(ts_neq) * FT(100)
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
) end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    u = state.ρu / state.ρ
    return abs(dot(nM, u))
end

@inline function flux_first_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)

    flux.ρq_tot = SVector(
        state.ρu[1] * state.ρq_tot / state.ρ,
        FT(0),
        state.ρu[3] * state.ρq_tot / state.ρ,
    )

    flux.ρe = SVector(
        state.ρu[1] / state.ρ * (state.ρe + aux.p),
        FT(0),
        state.ρu[3] / state.ρ * (state.ρe + aux.p),
    )

end

source!(::KinematicModel, _...) = nothing

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
    t_end = FT(60 * 30)
    dt = 40
    output_freq = 9

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

    mpicomm = MPI.COMM_WORLD

    vtkdir = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk"))
    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    model = driver_config.bl
    step = [0]
    cbvtk =
        GenericCallbacks.EveryXSimulationSteps(output_freq) do (init = false)
            out_dirname = @sprintf(
                "new_ex_1_mpirank%04d_step%04d",
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

    q_tot_ind = varsindex(vars_state_auxiliary(model, FT), :q_tot)
    q_vap_ind = varsindex(vars_state_auxiliary(model, FT), :q_vap)
    q_liq_ind = varsindex(vars_state_auxiliary(model, FT), :q_liq)
    q_ice_ind = varsindex(vars_state_auxiliary(model, FT), :q_ice)
    S_ind = varsindex(vars_state_auxiliary(model, FT), :S)

    result = ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cbvtk,),
        check_euclidean_distance = true,
    )

    max_S = maximum(abs.(solver_config.dg.state_auxiliary[:, S_ind, :]))
    @test isequal(max_S, FT(0))

    max_q_tot = maximum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    min_q_tot = minimum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    @test isapprox(max_q_tot, qt_0; rtol = 1e-3)
    @test isapprox(min_q_tot, qt_0; rtol = 1e-3)

    max_water_diff = maximum(abs.(
        solver_config.dg.state_auxiliary[:, q_tot_ind, :] .-
        solver_config.dg.state_auxiliary[:, q_vap_ind, :] .-
        solver_config.dg.state_auxiliary[:, q_liq_ind, :],
    ))
    @test isequal(max_water_diff, FT(0))

    max_q_ice = maximum(abs.(solver_config.dg.state_auxiliary[:, q_ice_ind, :]))
    @test isequal(max_q_ice, FT(0))

    max_q_liq = max(solver_config.dg.state_auxiliary[:, q_liq_ind, :]...)
    min_q_liq = min(solver_config.dg.state_auxiliary[:, q_liq_ind, :]...)
    @test max_q_liq < FT(1e-3)
    @test isequal(min_q_liq, FT(0))
end

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

