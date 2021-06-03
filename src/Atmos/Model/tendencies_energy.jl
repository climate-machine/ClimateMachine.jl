##### Energy tendencies

#####
##### First order fluxes
#####

function flux(::Energy, ::Advect, atmos, args)
    @unpack state = args
    return (state.ρu / state.ρ) * state.energy.ρe
end
function two_point_flux(
    ::AbstractKennedyGruberSplitForm,
    ::Energy,
    ::Advect,
    atmos,
    args,
)
    @unpack state1, state2 = args
    ρ1 = state1.ρ
    u1 = state1.ρu / ρ1
    e1 = state1.energy.ρe / ρ1

    ρ2 = state2.ρ
    u2 = state2.ρu / ρ2
    e2 = state2.energy.ρe / ρ2

    ρ_ave = (ρ1 + ρ2) / 2
    u_ave = (u1 + u2) / 2
    e_ave = (e1 + e2) / 2
    return ρ_ave * u_ave * e_ave
end

function flux(::ρθ_liq_ice, ::Advect, atmos, args)
    @unpack state = args
    return (state.ρu / state.ρ) * state.energy.ρθ_liq_ice
end

function flux(::Energy, ::Pressure, atmos, args)
    @unpack state = args
    @unpack ts = args.precomputed
    return state.ρu / state.ρ * air_pressure(ts)
end
function two_point_flux(
    ::AbstractKennedyGruberSplitForm,
    ::Energy,
    ::Pressure,
    atmos,
    args,
)
    @unpack state1, state2, aux1, aux2 = args
    ρ1 = state1.ρ
    u1 = state1.ρu / ρ1
    ts1 = new_thermo_state(atmos, state1, aux1)
    p1 = air_pressure(ts1)

    ρ2 = state2.ρ
    u2 = state2.ρu / ρ2
    ts2 = new_thermo_state(atmos, state2, aux2)
    p2 = air_pressure(ts2)

    u_ave = (u1 + u2) / 2
    p_ave = (p1 + p2) / 2
    return u_ave * p_ave
end

#####
##### Second order fluxes
#####

struct ViscousFlux <: TendencyDef{Flux{SecondOrder}} end
function flux(::Energy, ::ViscousFlux, atmos, args)
    @unpack state = args
    @unpack τ = args.precomputed.turbulence
    return τ * state.ρu
end

function flux(::ρθ_liq_ice, ::ViscousFlux, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    return state.ρ * (-D_t) .* diffusive.energy.∇θ_liq_ice
end

function flux(::Energy, ::HyperdiffViscousFlux, atmos, args)
    @unpack state, hyperdiffusive = args
    return hyperdiffusive.hyperdiffusion.ν∇³u_h * state.ρu
end

function flux(::Energy, ::HyperdiffEnthalpyFlux, atmos, args)
    @unpack state, hyperdiffusive = args
    return hyperdiffusive.hyperdiffusion.ν∇³h_tot * state.ρ
end

struct DiffEnthalpyFlux <: TendencyDef{Flux{SecondOrder}} end
function flux(::Energy, ::DiffEnthalpyFlux, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_h_tot = -D_t .* diffusive.energy.∇h_tot
    return d_h_tot * state.ρ
end

#####
##### Sources
#####

function source(::Energy, s::Subsidence, m, args)
    @unpack state, aux, diffusive = args
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.energy.∇h_tot)
end

function source(::Energy, s::RemovePrecipitation, m, args)
    @unpack state = args
    @unpack ts = args.precomputed
    if has_condensate(ts)
        nt = remove_precipitation_sources(s, m, args)
        return nt.S_ρ_e
    else
        FT = eltype(state)
        return FT(0)
    end
end

function source(::Energy, s::WarmRain_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_e
end

function source(::Energy, s::RainSnow_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_e
end
