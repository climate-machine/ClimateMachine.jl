##### Multi-physics types

using CLIMAParameters.Planet: T_freeze, cv_l

using ..Microphysics_0M
using ..Microphysics

export RainSnow_1M
export WarmRain_1M
export Diffusion
export Subsidence
export RemovePrecipitation

struct Subsidence{FT} <: TendencyDef{Source}
    D::FT
end

prognostic_vars(::Subsidence) = (Mass(), Energy(), TotalMoisture())

# Subsidence includes tendencies in Mass, Energy and TotalMoisture equations:

subsidence_velocity(subsidence::Subsidence{FT}, z::FT) where {FT} =
    -subsidence.D * z

struct PressureGradient <: TendencyDef{Flux{FirstOrder}} end
struct Pressure <: TendencyDef{Flux{FirstOrder}} end
struct Advect <: TendencyDef{Flux{FirstOrder}} end
struct GravityFluctuation <: TendencyDef{Flux{FirstOrder}} end
struct Diffusion <: TendencyDef{Flux{SecondOrder}} end
struct MoistureDiffusion <: TendencyDef{Flux{SecondOrder}} end

"""
    RemovePrecipitation <: TendencyDef{Source}

A sink to `q_tot` when cloud condensate is exceeding a threshold.
The threshold is defined either in terms of condensate or supersaturation.
The removal rate is implemented as a relaxation term
in the Microphysics_0M module.
The default thresholds and timescale are defined in CLIMAParameters.jl.
"""
struct RemovePrecipitation <: TendencyDef{Source}
    " Set to true if using qc based threshold"
    use_qc_thr::Bool
end

prognostic_vars(::RemovePrecipitation) = (Mass(), Energy(), TotalMoisture())

"""
    PrecipitationFlux <: TendencyDef{Flux{FirstOrder}}

Computes the precipitation flux as a sum of air velocity and terminal velocity
multiplied by the advected variable.
"""
struct PrecipitationFlux <: TendencyDef{Flux{FirstOrder}} end

function remove_precipitation_sources(s::RemovePrecipitation, atmos, args)
    @unpack state, aux = args
    @unpack ts = args.precomputed

    FT = eltype(state)

    q = PhasePartition(ts)
    λ::FT = liquid_fraction(ts)
    I_l::FT = internal_energy_liquid(ts)
    I_i::FT = internal_energy_ice(ts)
    Φ::FT = gravitational_potential(atmos.orientation, aux)

    S_qt::FT = 0
    param_set = parameter_set(atmos)
    if s.use_qc_thr
        S_qt = remove_precipitation(param_set, q)
    else
        q_vap_sat = q_vap_saturation(ts)
        S_qt = remove_precipitation(param_set, q, q_vap_sat)
    end

    S_e::FT = (λ * I_l + (1 - λ) * I_i + Φ) * S_qt

    return (; S_ρ_qt = state.ρ * S_qt, S_ρ_e = state.ρ * S_e)
end

"""
    WarmRain_1M <: TendencyDef{Source}

A collection of source/sink terms related to 1-moment warm rain microphysics.
"""
struct WarmRain_1M <: TendencyDef{Source} end

prognostic_vars(::WarmRain_1M) =
    (Mass(), Energy(), TotalMoisture(), LiquidMoisture(), Rain())

function warm_rain_sources(atmos, args, ts)
    @unpack state, aux = args

    FT = eltype(state)

    q = PhasePartition(ts)
    T::FT = air_temperature(ts)
    I_l::FT = internal_energy_liquid(ts)
    q_rai::FT = state.precipitation.ρq_rai / state.ρ
    Φ::FT = gravitational_potential(atmos.orientation, aux)
    param_set = parameter_set(atmos)

    # autoconversion
    src_q_rai_acnv = conv_q_liq_to_q_rai(param_set.microphys.rai, q.liq)
    # accretion
    src_q_rai_accr = accretion(
        param_set,
        param_set.microphys.liq,
        param_set.microphys.rai,
        q.liq,
        q_rai,
        state.ρ,
    )
    # rain evaporation
    src_q_rai_evap = evaporation_sublimation(
        param_set,
        param_set.microphys.rai,
        q,
        q_rai,
        state.ρ,
        T,
    )

    S_qr::FT = src_q_rai_acnv + src_q_rai_accr + src_q_rai_evap
    S_ql::FT = -src_q_rai_acnv - src_q_rai_accr
    S_qt::FT = -S_qr
    S_e::FT = S_qt * (I_l + Φ)

    return (;
        S_ρ_qt = state.ρ * S_qt,
        S_ρ_ql = state.ρ * S_ql,
        S_ρ_qr = state.ρ * S_qr,
        S_ρ_e = state.ρ * S_e,
    )
end

"""
    RainSnow_1M <: TendencyDef{Source}

A collection of source/sink terms related to 1-moment rain and snow microphysics
"""
struct RainSnow_1M <: TendencyDef{Source} end

prognostic_vars(::RainSnow_1M) = (
    Mass(),
    Energy(),
    TotalMoisture(),
    LiquidMoisture(),
    IceMoisture(),
    Rain(),
    Snow(),
)

function rain_snow_sources(atmos, args, ts)
    @unpack state, aux = args

    FT = eltype(state)
    param_set = parameter_set(atmos)

    q_rai::FT = state.precipitation.ρq_rai / state.ρ
    q_sno::FT = state.precipitation.ρq_sno / state.ρ
    q = PhasePartition(ts)
    T::FT = air_temperature(ts)
    I_d::FT = internal_energy_dry(ts)
    I_v::FT = internal_energy_vapor(ts)
    I_l::FT = internal_energy_liquid(ts)
    I_i::FT = internal_energy_ice(ts)
    _T_freeze::FT = T_freeze(param_set)
    _L_f::FT = latent_heat_fusion(ts)
    _cv_l::FT = cv_l(param_set)
    Φ::FT = gravitational_potential(atmos.orientation, aux)

    # temporary vars for summming different source terms
    S_qr::FT = FT(0)
    S_qs::FT = FT(0)
    S_ql::FT = FT(0)
    S_qi::FT = FT(0)
    S_qt::FT = FT(0)
    S_e::FT = FT(0)

    # source of rain via autoconversion
    tmp = conv_q_liq_to_q_rai(param_set.microphys.rai, q.liq)
    S_qr += tmp
    S_ql -= tmp
    S_e -= tmp * (I_l + Φ)

    # source of snow via autoconversion
    tmp = conv_q_ice_to_q_sno(param_set, param_set.microphys.ice, q, state.ρ, T)
    S_qs += tmp
    S_qi -= tmp
    S_e -= tmp * (I_i + Φ)

    # source of rain water via accretion cloud water - rain
    tmp = accretion(
        param_set,
        param_set.microphys.liq,
        param_set.microphys.rai,
        q.liq,
        q_rai,
        state.ρ,
    )
    S_qr += tmp
    S_ql -= tmp
    S_e -= tmp * (I_l + Φ)

    # source of snow via accretion cloud ice - snow
    tmp = accretion(
        param_set,
        param_set.microphys.ice,
        param_set.microphys.sno,
        q.ice,
        q_sno,
        state.ρ,
    )
    S_qs += tmp
    S_qi -= tmp
    S_e -= tmp * (I_i + Φ)

    # sink of cloud water via accretion cloud water - snow
    tmp = accretion(
        param_set,
        param_set.microphys.liq,
        param_set.microphys.sno,
        q.liq,
        q_sno,
        state.ρ,
    )
    if T < _T_freeze # cloud droplets freeze to become snow)
        S_qs += tmp
        S_ql -= tmp
        S_e -= tmp * (I_i + Φ)
    else # snow melts, both cloud water and snow become rain
        α::FT = _cv_l / _L_f * (T - _T_freeze)
        S_ql -= tmp
        S_qs -= tmp * α
        S_qr += tmp * (1 + α)
        S_e -= tmp * ((1 + α) * I_l - α * I_i + Φ)
    end

    # sink of cloud ice via accretion cloud ice - rain
    tmp1 = accretion(
        param_set,
        param_set.microphys.ice,
        param_set.microphys.rai,
        q.ice,
        q_rai,
        state.ρ,
    )
    # sink of rain via accretion cloud ice - rain
    tmp2 = accretion_rain_sink(
        param_set,
        param_set.microphys.ice,
        param_set.microphys.rai,
        q.ice,
        q_rai,
        state.ρ,
    )
    S_qi -= tmp1
    S_e -= tmp1 * (I_i + Φ)
    S_qr -= tmp2
    S_e += tmp2 * _L_f
    S_qs += tmp1 + tmp2

    # accretion rain - snow
    if T < _T_freeze
        tmp = accretion_snow_rain(
            param_set,
            param_set.microphys.sno,
            param_set.microphys.rai,
            q_sno,
            q_rai,
            state.ρ,
        )
        S_qs += tmp
        S_qr -= tmp
        S_e += tmp * _L_f
    else
        tmp = accretion_snow_rain(
            param_set,
            param_set.microphys.rai,
            param_set.microphys.sno,
            q_rai,
            q_sno,
            state.ρ,
        )
        S_qs -= tmp
        S_qr += tmp
        S_e -= tmp * _L_f
    end

    # rain evaporation sink (it already has negative sign for evaporation)
    tmp = evaporation_sublimation(
        param_set,
        param_set.microphys.rai,
        q,
        q_rai,
        state.ρ,
        T,
    )
    S_qr += tmp
    S_e -= tmp * (I_l + Φ)

    # snow sublimation/deposition source/sink
    tmp = evaporation_sublimation(
        param_set,
        param_set.microphys.sno,
        q,
        q_sno,
        state.ρ,
        T,
    )
    S_qs += tmp
    S_e -= tmp * (I_i + Φ)

    # snow melt
    tmp = snow_melt(param_set, param_set.microphys.sno, q_sno, state.ρ, T)
    S_qs -= tmp
    S_qr += tmp
    S_e -= tmp * _L_f

    # total qt sink is the sum of precip sources
    S_qt = -S_qr - S_qs

    return (;
        S_ρ_qt = state.ρ * S_qt,
        S_ρ_ql = state.ρ * S_ql,
        S_ρ_qi = state.ρ * S_qi,
        S_ρ_qr = state.ρ * S_qr,
        S_ρ_qs = state.ρ * S_qs,
        S_ρ_e = state.ρ * S_e,
    )
end
