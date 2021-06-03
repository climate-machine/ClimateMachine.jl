##### Moisture tendencies

export CreateClouds

#####
##### First order fluxes
#####

function flux(::TotalMoisture, ::Advect, atmos, args)
    @unpack state = args
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_tot
end

function two_point_flux(
    ::AbstractKennedyGruberSplitForm,
    ::TotalMoisture,
    ::Advect,
    atmos,
    args,
)
    @unpack state1, state2 = args
    ρ1 = state1.ρ
    u1 = state1.ρu / ρ1
    q1 = state1.moisture.ρq_tot / ρ1

    ρ2 = state2.ρ
    u2 = state2.ρu / ρ2
    q2 = state2.moisture.ρq_tot / ρ2

    ρ_ave = (ρ1 + ρ2) / 2
    u_ave = (u1 + u2) / 2
    q_ave = (q1 + q2) / 2
    return ρ_ave * u_ave * q_ave
end

function flux(::LiquidMoisture, ::Advect, atmos, args)
    @unpack state = args
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_liq
end

function flux(::IceMoisture, ::Advect, atmos, args)
    @unpack state = args
    u = state.ρu / state.ρ
    return u * state.moisture.ρq_ice
end

#####
##### Second order fluxes
#####

function flux(::TotalMoisture, ::MoistureDiffusion, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    return d_q_tot * state.ρ
end

function flux(::LiquidMoisture, ::MoistureDiffusion, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_q_liq = (-D_t) .* diffusive.moisture.∇q_liq
    return d_q_liq * state.ρ
end

function flux(::IceMoisture, ::MoistureDiffusion, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_q_ice = (-D_t) .* diffusive.moisture.∇q_ice
    return d_q_ice * state.ρ
end

function flux(::TotalMoisture, ::HyperdiffViscousFlux, atmos, args)
    @unpack state, hyperdiffusive = args
    return hyperdiffusive.hyperdiffusion.ν∇³q_tot * state.ρ
end

#####
##### Sources
#####

function source(::TotalMoisture, s::Subsidence, m, args)
    @unpack state, aux, diffusive = args
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end

"""
    CreateClouds <: TendencyDef{Source}

A source/sink to `q_liq` and `q_ice` implemented as a relaxation towards
equilibrium in the Microphysics module.
The default relaxation timescales are defined in CLIMAParameters.jl.
"""
struct CreateClouds <: TendencyDef{Source} end

prognostic_vars(::CreateClouds) = (LiquidMoisture(), IceMoisture())

function source(::LiquidMoisture, s::CreateClouds, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    # get current temperature and phase partition
    FT = eltype(state)
    q = PhasePartition(ts)
    T = air_temperature(ts)
    param_set = parameter_set(m)

    # phase partition corresponding to the current T and q.tot
    # (this is not the same as phase partition from saturation adjustment)
    ts_eq = PhaseEquil_ρTq(param_set, state.ρ, T, q.tot)
    q_eq = PhasePartition(ts_eq)

    # cloud condensate as relaxation source terms
    S_q_liq = conv_q_vap_to_q_liq_ice(param_set.microphys.liq, q_eq, q)

    return state.ρ * S_q_liq
end

function source(::IceMoisture, s::CreateClouds, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    # get current temperature and phase partition
    FT = eltype(state)
    q = PhasePartition(ts)
    T = air_temperature(ts)
    param_set = parameter_set(m)

    # phase partition corresponding to the current T and q.tot
    # (this is not the same as phase partition from saturation adjustment)
    ts_eq = PhaseEquil_ρTq(param_set, state.ρ, T, q.tot)
    q_eq = PhasePartition(ts_eq)

    # cloud condensate as relaxation source terms
    S_q_ice = conv_q_vap_to_q_liq_ice(param_set.microphys.ice, q_eq, q)

    return state.ρ * S_q_ice
end

function source(::TotalMoisture, s::RemovePrecipitation, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    if has_condensate(ts)
        nt = remove_precipitation_sources(s, m, args)
        return nt.S_ρ_qt
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(::TotalMoisture, s::WarmRain_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qt
end

function source(::LiquidMoisture, s::WarmRain_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_ql
end

function source(::TotalMoisture, s::RainSnow_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qt
end

function source(::LiquidMoisture, s::RainSnow_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_ql
end

function source(::IceMoisture, s::RainSnow_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qi
end
