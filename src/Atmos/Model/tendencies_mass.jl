##### Mass tendencies

#####
##### First order fluxes
#####

function flux(::Mass, ::Advect, atmos, args)
    return args.state.ρu
end
function two_point_flux(
    ::AbstractKennedyGruberSplitForm,
    ::Mass,
    ::Advect,
    atmos,
    args,
)
    @unpack state1, state2 = args
    ρ1 = state1.ρ
    u1 = state1.ρu / ρ1
    ρ2 = state2.ρ
    u2 = state2.ρu / ρ2
    ρ_ave = (ρ1 + ρ2) / 2
    u_ave = (u1 + u2) / 2
    return ρ_ave * u_ave
end

#####
##### Second order fluxes
#####

function flux(::Mass, ::MoistureDiffusion, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    return d_q_tot * state.ρ
end

#####
##### Sources
#####

function source(::Mass, s::Subsidence, m, args)
    @unpack state, aux, diffusive = args
    z = altitude(m, aux)
    w_sub = subsidence_velocity(s, z)
    k̂ = vertical_unit_vector(m, aux)
    return -state.ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
end

function source(::Mass, s::RemovePrecipitation, m, args)
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

function source(::Mass, s::WarmRain_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qt
end

function source(::Mass, s::RainSnow_1M, m, args)
    @unpack cache = args.precomputed.precipitation
    return cache.S_ρ_qt
end
