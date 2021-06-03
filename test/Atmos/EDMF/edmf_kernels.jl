#### EDMF model kernels

using CLIMAParameters.Planet: e_int_v0, grav, day, R_d, R_v, molmass_ratio
using Printf
using ClimateMachine.Atmos: nodal_update_auxiliary_state!, Advect

using ClimateMachine.BalanceLaws

using ClimateMachine.MPIStateArrays: MPIStateArray
using ClimateMachine.DGMethods: LocalGeometry, DGModel

import ClimateMachine.Mesh.Filters: vars_state_filtered

import ClimateMachine.BalanceLaws:
    vars_state,
    prognostic_vars,
    prognostic_to_primitive!,
    primitive_to_prognostic!,
    get_prog_state,
    get_specific_state,
    flux,
    precompute,
    source,
    eq_tends,
    update_auxiliary_state!,
    init_state_prognostic!,
    compute_gradient_argument!,
    compute_gradient_flux!

import ClimateMachine.TurbulenceConvection:
    init_aux_turbconv!,
    turbconv_nodal_update_auxiliary_state!,
    turbconv_boundary_state!,
    turbconv_normal_boundary_flux_second_order!

using ClimateMachine.Thermodynamics: air_pressure, air_density


include(joinpath("helper_funcs", "nondimensional_exchange_functions.jl"))
include(joinpath("helper_funcs", "lamb_smooth_minimum.jl"))
include(joinpath("helper_funcs", "utility_funcs.jl"))
include(joinpath("helper_funcs", "subdomain_statistics.jl"))
include(joinpath("helper_funcs", "diagnose_environment.jl"))
include(joinpath("helper_funcs", "subdomain_thermo_states.jl"))
include(joinpath("helper_funcs", "save_subdomain_temperature.jl"))
include(joinpath("closures", "entr_detr.jl"))
include(joinpath("closures", "pressure.jl"))
include(joinpath("closures", "mixing_length.jl"))
include(joinpath("closures", "turbulence_functions.jl"))
include(joinpath("closures", "surface_functions.jl"))


function vars_state(m::NTuple{N, Updraft}, st::Auxiliary, FT) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

vars_state(::Updraft, ::Auxiliary, FT) = @vars(T::FT)
vars_state(::Environment, ::Auxiliary, FT) = @vars(T::FT)

function vars_state(m::EDMF, st::Auxiliary, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT)
    )
end

function vars_state(::Updraft, ::Prognostic, FT)
    @vars(ρa::FT, ρaw::FT, ρaθ_liq::FT, ρaq_tot::FT,)
end

function vars_state(::Environment, ::Prognostic, FT)
    @vars(ρatke::FT, ρaθ_liq_cv::FT, ρaq_tot_cv::FT, ρaθ_liq_q_tot_cv::FT,)
end

function vars_state(::Updraft, ::Primitive, FT)
    @vars(a::FT, aw::FT, aθ_liq::FT, aq_tot::FT,)
end

function vars_state(::Environment, ::Primitive, FT)
    @vars(atke::FT, aθ_liq_cv::FT, aq_tot_cv::FT, aθ_liq_q_tot_cv::FT,)
end

function vars_state(
    m::NTuple{N, Updraft},
    st::Union{Prognostic, Primitive},
    FT,
) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

function vars_state(m::EDMF, st::Union{Prognostic, Primitive}, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT)
    )
end

function vars_state(::Updraft, ::Gradient, FT)
    @vars(w::FT,)
end

function vars_state(::Environment, ::Gradient, FT)
    @vars(
        θ_liq::FT,
        q_tot::FT,
        w::FT,
        tke::FT,
        θ_liq_cv::FT,
        q_tot_cv::FT,
        θ_liq_q_tot_cv::FT,
        θv::FT,
        h_tot::FT,
    )
end

function vars_state(m::NTuple{N, Updraft}, st::Gradient, FT) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

function vars_state(m::EDMF, st::Gradient, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT),
        u::FT,
        v::FT
    )
end

function vars_state(m::NTuple{N, Updraft}, st::GradientFlux, FT) where {N}
    return Tuple{ntuple(i -> vars_state(m[i], st, FT), N)...}
end

function vars_state(::Updraft, st::GradientFlux, FT)
    @vars(∇w::SVector{3, FT},)
end

function vars_state(::Environment, ::GradientFlux, FT)
    @vars(
        ∇θ_liq::SVector{3, FT},
        ∇q_tot::SVector{3, FT},
        ∇w::SVector{3, FT},
        ∇tke::SVector{3, FT},
        ∇θ_liq_cv::SVector{3, FT},
        ∇q_tot_cv::SVector{3, FT},
        ∇θ_liq_q_tot_cv::SVector{3, FT},
        ∇θv::SVector{3, FT},
        ∇h_tot::SVector{3, FT},
    )

end

function vars_state(m::EDMF, st::GradientFlux, FT)
    @vars(
        environment::vars_state(m.environment, st, FT),
        updraft::vars_state(m.updraft, st, FT),
        ∇u::SVector{3, FT},
        ∇v::SVector{3, FT}
    )
end

function vars_state_filtered(::Updraft, FT)
    @vars(a::FT, aw::FT, aθ_liq::FT, aq_tot::FT,)
end

function vars_state_filtered(m::NTuple{N, Updraft}, FT) where {N}
    return Tuple{ntuple(i -> vars_state_filtered(m[i], FT), N)...}
end

function vars_state_filtered(::Environment, FT)
    @vars(atke::FT, aθ_liq_cv::FT, aq_tot_cv::FT, aθ_liq_q_tot_cv::FT,)
end

function vars_state_filtered(m::EDMF, FT)
    @vars(
        environment::vars_state_filtered(m.environment, FT),
        updraft::vars_state_filtered(m.updraft, FT)
    )
end

abstract type EDMFPrognosticVariable <: AbstractPrognosticVariable end

abstract type EnvironmentPrognosticVariable <: EDMFPrognosticVariable end
struct en_ρatke <: EnvironmentPrognosticVariable end
struct en_ρaθ_liq_cv <: EnvironmentPrognosticVariable end
struct en_ρaq_tot_cv <: EnvironmentPrognosticVariable end
struct en_ρaθ_liq_q_tot_cv <: EnvironmentPrognosticVariable end

abstract type UpdraftPrognosticVariable{i} <: EDMFPrognosticVariable end
struct up_ρa{i} <: UpdraftPrognosticVariable{i} end
struct up_ρaw{i} <: UpdraftPrognosticVariable{i} end
struct up_ρaθ_liq{i} <: UpdraftPrognosticVariable{i} end
struct up_ρaq_tot{i} <: UpdraftPrognosticVariable{i} end

prognostic_vars(m::EDMF) =
    (prognostic_vars(m.environment)..., prognostic_vars(m.updraft)...)
prognostic_vars(m::Environment) =
    (en_ρatke(), en_ρaθ_liq_cv(), en_ρaq_tot_cv(), en_ρaθ_liq_q_tot_cv())

function prognostic_vars(m::NTuple{N, Updraft}) where {N}
    t_ρa = vuntuple(i -> up_ρa{i}(), N)
    t_ρaw = vuntuple(i -> up_ρaw{i}(), N)
    t_ρaθ_liq = vuntuple(i -> up_ρaθ_liq{i}(), N)
    t_ρaq_tot = vuntuple(i -> up_ρaq_tot{i}(), N)
    t = (t_ρa..., t_ρaw..., t_ρaθ_liq..., t_ρaq_tot...)
    return t
end

get_prog_state(state, ::en_ρatke) = (state.turbconv.environment, :ρatke)
get_prog_state(state, ::en_ρaθ_liq_cv) =
    (state.turbconv.environment, :ρaθ_liq_cv)
get_prog_state(state, ::en_ρaq_tot_cv) =
    (state.turbconv.environment, :ρaq_tot_cv)
get_prog_state(state, ::en_ρaθ_liq_q_tot_cv) =
    (state.turbconv.environment, :ρaθ_liq_q_tot_cv)

get_prog_state(state, ::up_ρa{i}) where {i} = (state.turbconv.updraft[i], :ρa)
get_prog_state(state, ::up_ρaw{i}) where {i} = (state.turbconv.updraft[i], :ρaw)
get_prog_state(state, ::up_ρaθ_liq{i}) where {i} =
    (state.turbconv.updraft[i], :ρaθ_liq)
get_prog_state(state, ::up_ρaq_tot{i}) where {i} =
    (state.turbconv.updraft[i], :ρaq_tot)

get_specific_state(state, ::en_ρatke) = (state.turbconv.environment, :atke)
get_specific_state(state, ::en_ρaθ_liq_cv) =
    (state.turbconv.environment, :aθ_liq_cv)
get_specific_state(state, ::en_ρaq_tot_cv) =
    (state.turbconv.environment, :aq_tot_cv)
get_specific_state(state, ::en_ρaθ_liq_q_tot_cv) =
    (state.turbconv.environment, :aθ_liq_q_tot_cv)

get_specific_state(state, ::up_ρa{i}) where {i} =
    (state.turbconv.updraft[i], :a)
get_specific_state(state, ::up_ρaw{i}) where {i} =
    (state.turbconv.updraft[i], :aw)
get_specific_state(state, ::up_ρaθ_liq{i}) where {i} =
    (state.turbconv.updraft[i], :aθ_liq)
get_specific_state(state, ::up_ρaq_tot{i}) where {i} =
    (state.turbconv.updraft[i], :aq_tot)

struct EntrDetr{N_up} <: TendencyDef{Source} end
struct PressSource{N_up} <: TendencyDef{Source} end
struct BuoySource{N_up} <: TendencyDef{Source} end
struct ShearSource <: TendencyDef{Source} end
struct DissSource <: TendencyDef{Source} end
struct GradProdSource <: TendencyDef{Source} end

prognostic_vars(::EntrDetr{N_up}) where {N_up} = (
    vuntuple(i -> up_ρa{i}, N_up)...,
    vuntuple(i -> up_ρaw{i}, N_up)...,
    vuntuple(i -> up_ρaθ_liq{i}, N_up)...,
    vuntuple(i -> up_ρaq_tot{i}, N_up)...,
    en_ρatke(),
    en_ρaθ_liq_cv(),
    en_ρaq_tot_cv(),
    en_ρaθ_liq_q_tot_cv(),
)
prognostic_vars(::PressSource{N_up}) where {N_up} =
    vuntuple(i -> up_ρaw{i}(), N_up)

prognostic_vars(::BuoySource{N_up}) where {N_up} =
    vuntuple(i -> up_ρaw{i}(), N_up)

EntrDetr(m::EDMF) = EntrDetr{n_updrafts(m)}()
BuoySource(m::EDMF) = BuoySource{n_updrafts(m)}()
PressSource(m::EDMF) = PressSource{n_updrafts(m)}()

# Dycore tendencies
eq_tends(
    pv::Union{Momentum, Energy, TotalMoisture},
    m::EDMF,
    flux::Flux{SecondOrder},
) = eq_tends(pv, m.coupling, flux)

eq_tends(
    pv::Union{Momentum, Energy, TotalMoisture},
    ::Decoupled,
    ::Flux{SecondOrder},
) = ()

eq_tends(
    pv::Union{Momentum, Energy, TotalMoisture},
    ::Coupled,
    ::Flux{SecondOrder},
) = (SGSFlux(),)

# Turbconv tendencies
eq_tends(pv::EDMFPrognosticVariable, m::AtmosModel, tt::Flux{O}) where {O} =
    eq_tends(pv, turbconv_model(m), tt)

eq_tends(::EDMFPrognosticVariable, m::EDMF, ::Flux{O}) where {O} = ()

eq_tends(::EnvironmentPrognosticVariable, m::EDMF, ::Flux{SecondOrder}) =
    (Diffusion(),)

eq_tends(pv::EDMFPrognosticVariable, m::EDMF, ::Flux{FirstOrder}) = (Advect(),)

eq_tends(pv::PV, m::EDMF, ::Source) where {PV} = ()

eq_tends(::EDMFPrognosticVariable, m::EDMF, ::Source) = (EntrDetr(m),)

eq_tends(pv::en_ρatke, m::EDMF, ::Source) =
    (EntrDetr(m), PressSource(m), BuoySource(m), ShearSource(), DissSource())

eq_tends(
    ::Union{en_ρaθ_liq_cv, en_ρaq_tot_cv, en_ρaθ_liq_q_tot_cv},
    m::EDMF,
    ::Source,
) = (EntrDetr(m), DissSource(), GradProdSource())

eq_tends(::up_ρaw, m::EDMF, ::Source) =
    (EntrDetr(m), PressSource(m), BuoySource(m))

struct SGSFlux <: TendencyDef{Flux{SecondOrder}} end

"""
    init_aux_turbconv!(
        turbconv::EDMF{FT},
        m::AtmosModel{FT},
        aux::Vars,
        geom::LocalGeometry,
    ) where {FT}

Initialize EDMF auxiliary variables.
"""
function init_aux_turbconv!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    aux::Vars,
    geom::LocalGeometry,
) where {FT} end;

function turbconv_nodal_update_auxiliary_state!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    save_subdomain_temperature!(m, state, aux)
end;

function prognostic_to_primitive!(
    turbconv::EDMF,
    atmos,
    moist::DryModel,
    prim::Vars,
    prog::Vars,
)
    N_up = n_updrafts(turbconv)
    prim_en = prim.turbconv.environment
    prog_en = prog.turbconv.environment
    prim_up = prim.turbconv.updraft
    prog_up = prog.turbconv.updraft

    ρ_inv = 1 / prog.ρ
    prim_en.atke = prog_en.ρatke * ρ_inv
    prim_en.aθ_liq_cv = prog_en.ρaθ_liq_cv * ρ_inv
    prim_en.aθ_liq_q_tot_cv = prog_en.ρaθ_liq_q_tot_cv * ρ_inv

    if moist isa DryModel
        prim_en.aq_tot_cv = 0
    else
        prim_en.aq_tot_cv = prog_en.ρaq_tot_cv * ρ_inv
    end
    @unroll_map(N_up) do i
        prim_up[i].a = prog_up[i].ρa * ρ_inv
        prim_up[i].aw = prog_up[i].ρaw * ρ_inv
        prim_up[i].aθ_liq = prog_up[i].ρaθ_liq * ρ_inv
        if moist isa DryModel
            prim_up[i].aq_tot = 0
        else
            prim_up[i].aq_tot = prog_up[i].ρaq_tot * ρ_inv
        end
    end
end

function primitive_to_prognostic!(
    turbconv::EDMF,
    atmos,
    moist::DryModel,
    prog::Vars,
    prim::Vars,
)

    N_up = n_updrafts(turbconv)
    prim_en = prim.turbconv.environment
    prog_en = prog.turbconv.environment
    prim_up = prim.turbconv.updraft
    prog_up = prog.turbconv.updraft

    ρ_gm = prog.ρ
    prog_en.ρatke = prim_en.atke * ρ_gm
    prog_en.ρaθ_liq_cv = prim_en.aθ_liq_cv * ρ_gm
    prog_en.ρaθ_liq_q_tot_cv = prim_en.aθ_liq_q_tot_cv * ρ_gm

    if moist isa DryModel
        prog_en.ρaq_tot_cv = 0
    else
        prog_en.ρaq_tot_cv = prim_en.aq_tot_cv * ρ_gm
    end
    @unroll_map(N_up) do i
        prog_up[i].ρa = prim_up[i].a * ρ_gm
        prog_up[i].ρaw = prim_up[i].aw * ρ_gm
        prog_up[i].ρaθ_liq = prim_up[i].aθ_liq * ρ_gm
        if moist isa DryModel
            prog_up[i].ρaq_tot = 0
        else
            prog_up[i].ρaq_tot = prim_up[i].aq_tot * ρ_gm
        end
    end

end

function compute_gradient_argument!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    N_up = n_updrafts(turbconv)
    z = altitude(m, aux)

    # Aliases:
    gm_tf = transform.turbconv
    up_tf = transform.turbconv.updraft
    en_tf = transform.turbconv.environment
    gm = state
    up = state.turbconv.updraft
    en = state.turbconv.environment

    # Recover thermo states
    ts = recover_thermo_state_all(m, state, aux)

    # Get environment variables
    env = environment_vars(state, N_up)
    param_set = parameter_set(m)
    @unroll_map(N_up) do i
        up_tf[i].w = fix_void_up(up[i].ρa, up[i].ρaw / up[i].ρa)
    end
    _grav::FT = grav(param_set)

    ρ_inv = 1 / gm.ρ
    θ_liq_en = liquid_ice_pottemp(ts.en)

    if moisture_model(m) isa DryModel
        q_tot_en = FT(0)
    else
        q_tot_en = total_specific_humidity(ts.en)
    end

    # populate gradient arguments
    en_tf.θ_liq = θ_liq_en
    en_tf.q_tot = q_tot_en
    en_tf.w = env.w

    en_tf.tke = en.ρatke / (env.a * gm.ρ)
    en_tf.θ_liq_cv = en.ρaθ_liq_cv / (env.a * gm.ρ)

    if moisture_model(m) isa DryModel
        en_tf.q_tot_cv = FT(0)
        en_tf.θ_liq_q_tot_cv = FT(0)
    else
        en_tf.q_tot_cv = en.ρaq_tot_cv / (env.a * gm.ρ)
        en_tf.θ_liq_q_tot_cv = en.ρaθ_liq_q_tot_cv / (env.a * gm.ρ)
    end

    en_tf.θv = virtual_pottemp(ts.en)
    en_e_kin =
        FT(1 // 2) * ((gm.ρu[1] * ρ_inv)^2 + (gm.ρu[2] * ρ_inv)^2 + env.w^2) # TBD: Check
    en_e_tot = total_energy(en_e_kin, _grav * z, ts.en)
    en_tf.h_tot = total_specific_enthalpy(ts.en, en_e_tot)

    gm_tf.u = gm.ρu[1] * ρ_inv
    gm_tf.v = gm.ρu[2] * ρ_inv
end;

function compute_gradient_flux!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT}
    args = (; diffusive, state, aux, t)
    N_up = n_updrafts(turbconv)

    # Aliases:
    gm = state
    gm_dif = diffusive.turbconv
    gm_∇tf = ∇transform.turbconv
    up_dif = diffusive.turbconv.updraft
    up_∇tf = ∇transform.turbconv.updraft
    en = state.turbconv.environment
    en_dif = diffusive.turbconv.environment
    en_∇tf = ∇transform.turbconv.environment

    @unroll_map(N_up) do i
        up_dif[i].∇w = up_∇tf[i].w
    end

    env = environment_vars(state, N_up)
    ρa₀ = gm.ρ * env.a
    # first moment grid mean coming from environment gradients only
    en_dif.∇θ_liq = en_∇tf.θ_liq
    en_dif.∇q_tot = en_∇tf.q_tot
    en_dif.∇w = en_∇tf.w
    # second moment env cov
    en_dif.∇tke = en_∇tf.tke
    en_dif.∇θ_liq_cv = en_∇tf.θ_liq_cv
    en_dif.∇q_tot_cv = en_∇tf.q_tot_cv
    en_dif.∇θ_liq_q_tot_cv = en_∇tf.θ_liq_q_tot_cv

    en_dif.∇θv = en_∇tf.θv
    en_dif.∇h_tot = en_∇tf.h_tot

    gm_dif.∇u = gm_∇tf.u
    gm_dif.∇v = gm_∇tf.v
end;

function source(::up_ρa{i}, ::EntrDetr, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, ρa_up = args.precomputed.turbconv
    return fix_void_up(ρa_up[i], E_dyn[i] - Δ_dyn[i])
end

function source(::up_ρaw{i}, ::EntrDetr, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, E_trb, env, ρa_up, w_up = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    entr = fix_void_up(ρa_up[i], (E_dyn[i] + E_trb[i]) * env.w)
    detr = fix_void_up(ρa_up[i], (Δ_dyn[i] + E_trb[i]) * w_up[i])

    return entr - detr
end

function source(::up_ρaθ_liq{i}, ::EntrDetr, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, E_trb, env, ρa_up, ts_en = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    θ_liq_en = liquid_ice_pottemp(ts_en)
    entr = fix_void_up(ρa_up[i], (E_dyn[i] + E_trb[i]) * θ_liq_en)
    detr =
        fix_void_up(ρa_up[i], (Δ_dyn[i] + E_trb[i]) * up[i].ρaθ_liq / ρa_up[i])

    return entr - detr
end

function source(::up_ρaq_tot{i}, ::EntrDetr, atmos, args) where {i}
    @unpack E_dyn, Δ_dyn, E_trb, ρa_up, ts_en = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    q_tot_en = total_specific_humidity(ts_en)
    entr = fix_void_up(ρa_up[i], (E_dyn[i] + E_trb[i]) * q_tot_en)
    detr =
        fix_void_up(ρa_up[i], (Δ_dyn[i] + E_trb[i]) * up[i].ρaq_tot / ρa_up[i])

    return entr - detr
end

function source(::en_ρatke, ::EntrDetr, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, env, ρa_up, w_up = args.precomputed.turbconv
    @unpack state = args
    up = state.turbconv.updraft
    en = state.turbconv.environment
    gm = state
    N_up = n_updrafts(turbconv_model(atmos))
    ρ_inv = 1 / gm.ρ
    tke_en = enforce_positivity(en.ρatke) * ρ_inv / env.a

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            E_trb[i] * (env.w - gm.ρu[3] * ρ_inv) * (env.w - w_up[i]) -
            (E_dyn[i] + E_trb[i]) * tke_en +
            Δ_dyn[i] * (w_up[i] - env.w) * (w_up[i] - env.w) / 2,
        )
    end
    return sum(entr_detr)
end

function source(::en_ρaθ_liq_cv, ::EntrDetr, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, ρa_up, ts_en = args.precomputed.turbconv
    @unpack state = args
    ts_gm = args.precomputed.ts
    up = state.turbconv.updraft
    en = state.turbconv.environment
    N_up = n_updrafts(turbconv_model(atmos))
    θ_liq = liquid_ice_pottemp(ts_gm)
    θ_liq_en = liquid_ice_pottemp(ts_en)

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            Δ_dyn[i] *
            (up[i].ρaθ_liq / ρa_up[i] - θ_liq_en) *
            (up[i].ρaθ_liq / ρa_up[i] - θ_liq_en) +
            E_trb[i] *
            (θ_liq_en - θ_liq) *
            (θ_liq_en - up[i].ρaθ_liq / ρa_up[i]) +
            E_trb[i] *
            (θ_liq_en - θ_liq) *
            (θ_liq_en - up[i].ρaθ_liq / ρa_up[i]) -
            (E_dyn[i] + E_trb[i]) * en.ρaθ_liq_cv,
        )
    end
    return sum(entr_detr)
end

function source(::en_ρaq_tot_cv, ::EntrDetr, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, ρa_up, ts_en = args.precomputed.turbconv
    @unpack state = args
    FT = eltype(state)
    up = state.turbconv.updraft
    en = state.turbconv.environment
    gm = state
    N_up = n_updrafts(turbconv_model(atmos))
    q_tot_en = total_specific_humidity(ts_en)
    ρ_inv = 1 / gm.ρ
    ρq_tot = moisture_model(atmos) isa DryModel ? FT(0) : gm.moisture.ρq_tot

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            Δ_dyn[i] *
            (up[i].ρaq_tot / ρa_up[i] - q_tot_en) *
            (up[i].ρaq_tot / ρa_up[i] - q_tot_en) +
            E_trb[i] *
            (q_tot_en - ρq_tot * ρ_inv) *
            (q_tot_en - up[i].ρaq_tot / ρa_up[i]) +
            E_trb[i] *
            (q_tot_en - ρq_tot * ρ_inv) *
            (q_tot_en - up[i].ρaq_tot / ρa_up[i]) -
            (E_dyn[i] + E_trb[i]) * en.ρaq_tot_cv,
        )
    end
    return sum(entr_detr)
end

function source(::en_ρaθ_liq_q_tot_cv, ::EntrDetr, atmos, args)
    @unpack E_dyn, Δ_dyn, E_trb, ρa_up, ts_en = args.precomputed.turbconv
    @unpack state = args
    FT = eltype(state)
    ts_gm = args.precomputed.ts
    up = state.turbconv.updraft
    en = state.turbconv.environment
    gm = state
    N_up = n_updrafts(turbconv_model(atmos))
    q_tot_en = total_specific_humidity(ts_en)
    θ_liq = liquid_ice_pottemp(ts_gm)
    θ_liq_en = liquid_ice_pottemp(ts_en)
    ρ_inv = 1 / gm.ρ
    ρq_tot = moisture_model(atmos) isa DryModel ? FT(0) : gm.moisture.ρq_tot

    entr_detr = vuntuple(N_up) do i
        fix_void_up(
            ρa_up[i],
            Δ_dyn[i] *
            (up[i].ρaθ_liq / ρa_up[i] - θ_liq_en) *
            (up[i].ρaq_tot / ρa_up[i] - q_tot_en) +
            E_trb[i] *
            (θ_liq_en - θ_liq) *
            (q_tot_en - up[i].ρaq_tot / ρa_up[i]) +
            E_trb[i] *
            (q_tot_en - ρq_tot * ρ_inv) *
            (θ_liq_en - up[i].ρaθ_liq / ρa_up[i]) -
            (E_dyn[i] + E_trb[i]) * en.ρaθ_liq_q_tot_cv,
        )
    end
    return sum(entr_detr)
end

function source(::en_ρatke, ::PressSource, atmos, args)
    @unpack env, ρa_up, dpdz, w_up = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    N_up = n_updrafts(turbconv_model(atmos))
    press_tke = vuntuple(N_up) do i
        fix_void_up(ρa_up[i], ρa_up[i] * (w_up[i] - env.w) * dpdz[i])
    end
    return sum(press_tke)
end

function source(::en_ρatke, ::ShearSource, atmos, args)
    @unpack env, K_m, Shear² = args.precomputed.turbconv
    gm = args.state
    ρa₀ = gm.ρ * env.a
    # production from mean gradient and Dissipation
    return ρa₀ * K_m * Shear² # tke Shear source
end

function source(::en_ρatke, ::BuoySource, atmos, args)
    @unpack env, K_h, ∂b∂z_env = args.precomputed.turbconv
    gm = args.state
    ρa₀ = gm.ρ * env.a
    return -ρa₀ * K_h * ∂b∂z_env   # tke Buoyancy source
end

function source(::en_ρatke, ::DissSource, atmos, args)
    @unpack Diss₀ = args.precomputed.turbconv
    en = args.state.turbconv.environment
    return -Diss₀ * en.ρatke  # tke Dissipation
end

function source(::en_ρaθ_liq_cv, ::DissSource, atmos, args)
    @unpack Diss₀ = args.precomputed.turbconv
    en = args.state.turbconv.environment
    return -Diss₀ * en.ρaθ_liq_cv
end

function source(::en_ρaq_tot_cv, ::DissSource, atmos, args)
    @unpack Diss₀ = args.precomputed.turbconv
    en = args.state.turbconv.environment
    return -Diss₀ * en.ρaq_tot_cv
end

function source(::en_ρaθ_liq_q_tot_cv, ::DissSource, atmos, args)
    @unpack Diss₀ = args.precomputed.turbconv
    en = args.state.turbconv.environment
    return -Diss₀ * en.ρaθ_liq_q_tot_cv
end

function source(::en_ρaθ_liq_cv, ::GradProdSource, atmos, args)
    @unpack env, K_h = args.precomputed.turbconv
    gm = args.state
    en_dif = args.diffusive.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return ρa₀ * (2 * K_h * en_dif.∇θ_liq[3] * en_dif.∇θ_liq[3])
end

function source(::en_ρaq_tot_cv, ::GradProdSource, atmos, args)
    @unpack env, K_h = args.precomputed.turbconv
    gm = args.state
    en_dif = args.diffusive.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return ρa₀ * (2 * K_h * en_dif.∇q_tot[3] * en_dif.∇q_tot[3])
end

function source(::en_ρaθ_liq_q_tot_cv, ::GradProdSource, atmos, args)
    @unpack env, K_h = args.precomputed.turbconv
    gm = args.state
    en_dif = args.diffusive.turbconv.environment
    ρa₀ = gm.ρ * env.a
    return ρa₀ * (2 * K_h * en_dif.∇θ_liq[3] * en_dif.∇q_tot[3])
end

function source(::up_ρaw{i}, ::BuoySource, atmos, args) where {i}
    @unpack buoy = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    return up[i].ρa * buoy.up[i]
end

function source(::up_ρaw{i}, ::PressSource, atmos, args) where {i}
    @unpack dpdz = args.precomputed.turbconv
    up = args.state.turbconv.updraft
    return -up[i].ρa * dpdz[i]
end

function compute_ρa_up(atmos, state, aux)
    # Aliases:
    turbconv = turbconv_model(atmos)
    gm = state
    up = state.turbconv.updraft
    N_up = n_updrafts(turbconv)
    a_min = turbconv.subdomains.a_min
    a_max = turbconv.subdomains.a_max
    # in future GCM implementations we need to think about grid mean advection
    ρa_up = vuntuple(N_up) do i
        gm.ρ * enforce_unit_bounds(up[i].ρa / gm.ρ, a_min, a_max)
    end
    return ρa_up
end

function flux(::up_ρa{i}, ::Advect, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], up[i].ρaw) * ẑ
end
function flux(::up_ρaw{i}, ::Advect, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up, w_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], up[i].ρaw * w_up[i]) * ẑ

end
function flux(::up_ρaθ_liq{i}, ::Advect, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up, w_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], w_up[i] * up[i].ρaθ_liq) * ẑ

end
function flux(::up_ρaq_tot{i}, ::Advect, atmos, args) where {i}
    @unpack state, aux = args
    @unpack ρa_up, w_up = args.precomputed.turbconv
    up = state.turbconv.updraft
    ẑ = vertical_unit_vector(atmos, aux)
    return fix_void_up(ρa_up[i], w_up[i] * up[i].ρaq_tot) * ẑ

end

function flux(::en_ρatke, ::Advect, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρatke * env.w * ẑ
end
function flux(::en_ρaθ_liq_cv, ::Advect, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρaθ_liq_cv * env.w * ẑ
end
function flux(::en_ρaq_tot_cv, ::Advect, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρaq_tot_cv * env.w * ẑ
end
function flux(::en_ρaθ_liq_q_tot_cv, ::Advect, atmos, args)
    @unpack state, aux = args
    @unpack env = args.precomputed.turbconv
    en = state.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return en.ρaθ_liq_q_tot_cv * env.w * ẑ
end

function precompute(::EDMF, bl, args, ts, ::Flux{FirstOrder})
    @unpack state, aux = args
    FT = eltype(state)
    turbconv = turbconv_model(bl)
    env = environment_vars(state, n_updrafts(turbconv))
    ρa_up = compute_ρa_up(bl, state, aux)
    up = state.turbconv.updraft
    gm = state
    N_up = n_updrafts(turbconv)
    ρ_inv = 1 / gm.ρ
    w_up = vuntuple(N_up) do i
        fix_void_up(ρa_up[i], up[i].ρaw / ρa_up[i])
    end

    θ_liq_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρaθ_liq / up[i].ρa, liquid_ice_pottemp(ts))
    end
    a_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρa * ρ_inv)
    end
    if !(moisture_model(bl) isa DryModel)
        q_tot_up = vuntuple(N_up) do i
            fix_void_up(up[i].ρa, up[i].ρaq_tot / up[i].ρa, gm.moisture.ρq_tot)
        end
    else
        q_tot_up = vuntuple(i -> FT(0), N_up)
    end

    return (; env, a_up, q_tot_up, ρa_up, θ_liq_up, w_up)
end

function precompute(::EDMF, bl, args, ts, ::Flux{SecondOrder})
    @unpack state, aux, diffusive, t = args
    en_dif = diffusive.turbconv.environment
    up = state.turbconv.updraft
    gm = state
    ts_gm = ts
    FT = eltype(state)
    z = altitude(bl, aux)
    param_set = parameter_set(bl)
    turbconv = turbconv_model(bl)
    N_up = n_updrafts(turbconv)
    _grav::FT = grav(param_set)
    ρ_inv = 1 / gm.ρ

    env = environment_vars(state, N_up)
    ts_en = new_thermo_state_en(bl, moisture_model(bl), state, aux, ts_gm)
    ts_up = new_thermo_state_up(bl, moisture_model(bl), state, aux, ts_gm)

    buoy = compute_buoyancy(bl, state, env, ts_en, ts_up, aux.ref_state)

    E_dyn, Δ_dyn, E_trb = entr_detr(bl, state, aux, ts_up, ts_en, env, buoy)

    Shear² =
        diffusive.turbconv.∇u[3]^2 +
        diffusive.turbconv.∇v[3]^2 +
        diffusive.turbconv.environment.∇w[3]^2
    l_mix, ∂b∂z_env, Pr_t = mixing_length(
        bl,
        turbconv.mix_len,
        args,
        Δ_dyn,
        E_trb,
        Shear²,
        ts_gm,
        ts_en,
        env,
    )
    ρa_up = compute_ρa_up(bl, state, aux)

    en = state.turbconv.environment
    tke_en = enforce_positivity(en.ρatke) / env.a / state.ρ
    K_m = turbconv.mix_len.c_m * l_mix * sqrt(tke_en)
    K_h = K_m / Pr_t
    ρaw_up = vuntuple(i -> up[i].ρaw, N_up)

    w_up = vuntuple(N_up) do i
        fix_void_up(ρa_up[i], up[i].ρaw / ρa_up[i])
    end

    ρu_gm_tup = Tuple(gm.ρu)
    ρa_en = gm.ρ * env.a
    # TODO: Consider turbulent contribution:

    e_kin_up =
        FT(1 / 2) .* ntuple(N_up) do i
            (ρu_gm_tup[1] * ρ_inv)^2 + (ρu_gm_tup[2] * ρ_inv)^2 + w_up[i]^2
        end
    e_kin_en =
        FT(1 // 2) * ((gm.ρu[1] * ρ_inv)^2 + (gm.ρu[2] * ρ_inv)^2 + env.w)^2

    e_tot_up = ntuple(i -> total_energy(e_kin_up[i], _grav * z, ts_up[i]), N_up)
    h_tot_up = ntuple(i -> total_specific_enthalpy(ts_up[i], e_tot_up[i]), N_up)

    e_tot_en = total_energy(e_kin_en, _grav * z, ts_en)
    h_tot_en = total_specific_enthalpy(ts_en, e_tot_en)
    h_tot_gm = total_specific_enthalpy(ts, gm.energy.ρe * ρ_inv)

    massflux_h_tot = sum(
        ntuple(N_up) do i
            fix_void_up(
                ρa_up[i],
                ρa_up[i] *
                (h_tot_gm - h_tot_up[i]) *
                (gm.ρu[3] * ρ_inv - ρaw_up[i] / ρa_up[i]),
            )
        end,
    )
    massflux_h_tot +=
        (ρa_en * (h_tot_gm - h_tot_en) * (ρu_gm_tup[3] * ρ_inv - env.w))

    ρh_sgs_flux = SVector{3, FT}(
        0,
        0,
        -gm.ρ * env.a * K_h * en_dif.∇h_tot[3] + massflux_h_tot,
    )

    return (;
        env,
        ρa_up,
        ρaw_up,
        ts_en,
        ts_up,
        E_dyn,
        Δ_dyn,
        E_trb,
        l_mix,
        ∂b∂z_env,
        K_h,
        K_m,
        Pr_t,
        ρh_sgs_flux,
    )
end

"""
    compute_buoyancy(
        bl::BalanceLaw,
        state::Vars,
        env::NamedTuple,
        ts_en::ThermodynamicState,
        ts_up,
        ref_state::Vars
    )

Compute buoyancies of subdomains
"""
function compute_buoyancy(
    bl::BalanceLaw,
    state::Vars,
    env::NamedTuple,
    ts_en::ThermodynamicState,
    ts_up,
    ref_state::Vars,
)
    FT = eltype(state)
    param_set = parameter_set(bl)
    N_up = n_updrafts(turbconv_model(bl))
    _grav::FT = grav(param_set)
    gm = state
    ρ_inv = 1 / gm.ρ
    buoyancy_en = -_grav * (air_density(ts_en) - ref_state.ρ) * ρ_inv
    up = state.turbconv.updraft

    a_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρa * ρ_inv)
    end

    abs_buoyancy_up = vuntuple(N_up) do i
        -_grav * (air_density(ts_up[i]) - ref_state.ρ) * ρ_inv
    end
    b_gm = grid_mean_b(env, a_up, N_up, abs_buoyancy_up, buoyancy_en)

    # remove the gm_b from all subdomains
    buoyancy_up = vuntuple(N_up) do i
        abs_buoyancy_up[i] - b_gm
    end

    buoyancy_en -= b_gm
    return (; up = buoyancy_up, en = buoyancy_en)
end

function precompute(::EDMF, bl, args, ts, ::Source)
    @unpack state, aux, diffusive, t = args
    ts_gm = ts
    gm = state
    up = state.turbconv.updraft
    turbconv = turbconv_model(bl)
    N_up = n_updrafts(turbconv)
    # Get environment variables
    env = environment_vars(state, N_up)
    # Recover thermo states
    ts_en = new_thermo_state_en(bl, moisture_model(bl), state, aux, ts_gm)
    ts_up = new_thermo_state_up(bl, moisture_model(bl), state, aux, ts_gm)
    ρa_up = compute_ρa_up(bl, state, aux)

    Shear² =
        diffusive.turbconv.∇u[3]^2 +
        diffusive.turbconv.∇v[3]^2 +
        diffusive.turbconv.environment.∇w[3]^2

    buoy = compute_buoyancy(bl, state, env, ts_en, ts_up, aux.ref_state)
    E_dyn, Δ_dyn, E_trb = entr_detr(bl, state, aux, ts_up, ts_en, env, buoy)

    dpdz = perturbation_pressure(bl, args, env, buoy)

    l_mix, ∂b∂z_env, Pr_t = mixing_length(
        bl,
        turbconv.mix_len,
        args,
        Δ_dyn,
        E_trb,
        Shear²,
        ts_gm,
        ts_en,
        env,
    )

    w_up = vuntuple(N_up) do i
        fix_void_up(ρa_up[i], up[i].ρaw / ρa_up[i])
    end

    en = state.turbconv.environment
    tke_en = enforce_positivity(en.ρatke) / env.a / state.ρ
    K_m = turbconv.mix_len.c_m * l_mix * sqrt(tke_en)
    K_h = K_m / Pr_t
    Diss₀ = turbconv.mix_len.c_d * sqrt(tke_en) / l_mix
    tke_buoy_prod = -gm.ρ * env.a * K_h * ∂b∂z_env   # tke Buoyancy source

    return (;
        env,
        Diss₀,
        buoy,
        K_m,
        K_h,
        ρa_up,
        w_up,
        ts_en,
        ts_up,
        E_dyn,
        Δ_dyn,
        E_trb,
        dpdz,
        l_mix,
        ∂b∂z_env,
        Pr_t,
        Shear²,
        tke_buoy_prod,
    )
end

function flux(::Energy, ::SGSFlux, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack ts = args.precomputed
    @unpack ρh_sgs_flux = args.precomputed.turbconv
    return ρh_sgs_flux
end

function flux(::TotalMoisture, ::SGSFlux, atmos, args)
    @unpack state, diffusive = args
    @unpack env, K_h, ρa_up, ρaw_up, ts_en = args.precomputed.turbconv
    FT = eltype(state)
    en_dif = diffusive.turbconv.environment
    up = state.turbconv.updraft
    gm = state
    ρ_inv = 1 / gm.ρ
    N_up = n_updrafts(turbconv_model(atmos))
    ρq_tot = moisture_model(atmos) isa DryModel ? FT(0) : gm.moisture.ρq_tot
    ρaq_tot_up = vuntuple(i -> up[i].ρaq_tot, N_up)
    ρa_en = gm.ρ * env.a
    q_tot_en = total_specific_humidity(ts_en)

    ρu_gm_tup = Tuple(gm.ρu)

    massflux_q_tot = sum(
        ntuple(N_up) do i
            fix_void_up(
                ρa_up[i],
                ρa_up[i] *
                (ρq_tot * ρ_inv - ρaq_tot_up[i] / ρa_up[i]) *
                (ρu_gm_tup[3] * ρ_inv - ρaw_up[i] / ρa_up[i]),
            )
        end,
    )
    massflux_q_tot +=
        (ρa_en * (ρq_tot * ρ_inv - q_tot_en) * (ρu_gm_tup[3] * ρ_inv - env.w))
    ρq_tot_sgs_flux = -gm.ρ * env.a * K_h * en_dif.∇q_tot[3] + massflux_q_tot
    return SVector{3, FT}(0, 0, ρq_tot_sgs_flux)
end

function flux(::Momentum, ::SGSFlux, atmos, args)
    @unpack state, diffusive = args
    @unpack env, K_m, ρa_up, ρaw_up = args.precomputed.turbconv
    FT = eltype(state)
    en_dif = diffusive.turbconv.environment
    gm_dif = diffusive.turbconv
    up = state.turbconv.updraft
    gm = state
    ρ_inv = 1 / gm.ρ
    N_up = n_updrafts(turbconv_model(atmos))
    ρa_en = gm.ρ * env.a

    ρu_gm_tup = Tuple(gm.ρu)

    massflux_w = sum(
        ntuple(N_up) do i
            fix_void_up(
                ρa_up[i],
                ρa_up[i] *
                (ρu_gm_tup[3] * ρ_inv - ρaw_up[i] / ρa_up[i]) *
                (ρu_gm_tup[3] * ρ_inv - ρaw_up[i] / ρa_up[i]),
            )
        end,
    )
    massflux_w += (
        ρa_en * (ρu_gm_tup[3] * ρ_inv - env.w) * (ρu_gm_tup[3] * ρ_inv - env.w)
    )
    ρw_sgs_flux = -gm.ρ * env.a * K_m * en_dif.∇w[3] + massflux_w
    ρu_sgs_flux = -gm.ρ * env.a * K_m * gm_dif.∇u[3]
    ρv_sgs_flux = -gm.ρ * env.a * K_m * gm_dif.∇v[3]
    return SMatrix{3, 3, FT, 9}(
        0,
        0,
        ρu_sgs_flux,
        0,
        0,
        ρv_sgs_flux,
        0,
        0,
        ρw_sgs_flux,
    )
end

function flux(::en_ρaθ_liq_cv, ::Diffusion, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, l_mix, Pr_t, K_h = args.precomputed.turbconv
    en_dif = diffusive.turbconv.environment
    gm = state
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_h * en_dif.∇θ_liq_cv[3] * ẑ
end
function flux(::en_ρaq_tot_cv, ::Diffusion, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, l_mix, Pr_t, K_h = args.precomputed.turbconv
    en_dif = diffusive.turbconv.environment
    gm = state
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_h * en_dif.∇q_tot_cv[3] * ẑ
end
function flux(::en_ρaθ_liq_q_tot_cv, ::Diffusion, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, l_mix, Pr_t, K_h = args.precomputed.turbconv
    en_dif = diffusive.turbconv.environment
    gm = state
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_h * en_dif.∇θ_liq_q_tot_cv[3] * ẑ
end
function flux(::en_ρatke, ::Diffusion, atmos, args)
    @unpack state, aux, diffusive = args
    @unpack env, K_m = args.precomputed.turbconv
    gm = state
    en_dif = diffusive.turbconv.environment
    ẑ = vertical_unit_vector(atmos, aux)
    return -gm.ρ * env.a * K_m * en_dif.∇tke[3] * ẑ
end

# First order boundary conditions
function turbconv_boundary_state!(
    nf,
    bc::EDMFBottomBC,
    atmos::AtmosModel{FT},
    state⁺::Vars,
    args,
) where {FT}
    @unpack state⁻, aux⁻, aux_int⁻ = args
    turbconv = turbconv_model(atmos)
    N_up = n_updrafts(turbconv)
    up⁺ = state⁺.turbconv.updraft
    en⁺ = state⁺.turbconv.environment
    gm⁻ = state⁻
    gm_a⁻ = aux⁻

    zLL = altitude(atmos, aux_int⁻)
    surf_vals = subdomain_surface_values(atmos, gm⁻, gm_a⁻, zLL)
    a_up_surf = surf_vals.a_up_surf

    @unroll_map(N_up) do i
        up⁺[i].ρaw = FT(0)
        up⁺[i].ρa = gm⁻.ρ * a_up_surf[i]
        up⁺[i].ρaθ_liq = gm⁻.ρ * a_up_surf[i] * surf_vals.θ_liq_up_surf[i]
        if !(moisture_model(atmos) isa DryModel)
            up⁺[i].ρaq_tot = gm⁻.ρ * a_up_surf[i] * surf_vals.q_tot_up_surf[i]
        else
            up⁺[i].ρaq_tot = FT(0)
        end
    end

    a_en = environment_area(gm⁻, N_up)
    en⁺.ρatke = gm⁻.ρ * a_en * surf_vals.tke
    en⁺.ρaθ_liq_cv = gm⁻.ρ * a_en * surf_vals.θ_liq_cv
    if !(moisture_model(atmos) isa DryModel)
        en⁺.ρaq_tot_cv = gm⁻.ρ * a_en * surf_vals.q_tot_cv
        en⁺.ρaθ_liq_q_tot_cv = gm⁻.ρ * a_en * surf_vals.θ_liq_q_tot_cv
    else
        en⁺.ρaq_tot_cv = FT(0)
        en⁺.ρaθ_liq_q_tot_cv = FT(0)
    end
end;
function turbconv_boundary_state!(
    nf,
    bc::EDMFTopBC,
    atmos::AtmosModel{FT},
    state⁺::Vars,
    args,
) where {FT}
    N_up = n_updrafts(turbconv_model(atmos))
    up⁺ = state⁺.turbconv.updraft
    @unroll_map(N_up) do i
        up⁺[i].ρaw = FT(0)
    end
end;


# The boundary conditions for second-order unknowns
# (here we prescribe a flux at state⁺ to match that at state⁻ so that the flux divergence is zero)
function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc::EDMFBottomBC,
    atmos::AtmosModel,
    fluxᵀn::Vars,
    args,
)
    @unpack state⁻, aux⁻, diffusive⁻, hyperdiff⁻, t, n⁻ = args
    en_flx = fluxᵀn.turbconv.environment
    tend_type = Flux{SecondOrder}()
    _args⁻ = (;
        state = state⁻,
        aux = aux⁻,
        t,
        diffusive = diffusive⁻,
        hyperdiffusive = hyperdiff⁻,
    )
    pargs = merge(_args⁻, (precomputed = precompute(atmos, _args⁻, tend_type),))

    total_flux = Σfluxes(
        en_ρatke(),
        eq_tends(en_ρatke(), atmos, tend_type),
        atmos,
        pargs,
    )
    nd_ρatke = dot(n⁻, total_flux)
    en_flx.ρatke = nd_ρatke

    total_flux = Σfluxes(
        en_ρaθ_liq_cv(),
        eq_tends(en_ρaθ_liq_cv(), atmos, tend_type),
        atmos,
        pargs,
    )
    nd_ρaθ_liq_cv = dot(n⁻, total_flux)
    en_flx.ρaθ_liq_cv = nd_ρaθ_liq_cv

    if !(moisture_model(atmos) isa DryModel)
        total_flux = Σfluxes(
            en_ρaq_tot_cv(),
            eq_tends(en_ρaq_tot_cv(), atmos, tend_type),
            atmos,
            pargs,
        )
        nd_ρaq_tot_cv = dot(n⁻, total_flux)
        en_flx.ρaq_tot_cv = nd_ρaq_tot_cv

        total_flux = Σfluxes(
            en_ρaθ_liq_q_tot_cv(),
            eq_tends(en_ρaθ_liq_q_tot_cv(), atmos, tend_type),
            atmos,
            pargs,
        )
        nd_ρaθ_liq_q_tot_cv = dot(n⁻, total_flux)
        en_flx.ρaθ_liq_q_tot_cv = nd_ρaθ_liq_q_tot_cv
    end
end;

function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc::EDMFTopBC,
    atmos::AtmosModel{FT},
    fluxᵀn::Vars,
    args,
) where {FT}

    turbconv = turbconv_model(atmos)
    N_up = n_updrafts(turbconv)
    up_flx = fluxᵀn.turbconv.updraft
    en_flx = fluxᵀn.turbconv.environment
    @unroll_map(N_up) do i
        up_flx[i].ρa = FT(0)
        up_flx[i].ρaθ_liq = FT(0)
        up_flx[i].ρaq_tot = FT(0)
    end
    en_flx.ρatke = FT(0)
    en_flx.ρaθ_liq_cv = FT(0)
    en_flx.ρaq_tot_cv = FT(0)
    en_flx.ρaθ_liq_q_tot_cv = FT(0)

end;
