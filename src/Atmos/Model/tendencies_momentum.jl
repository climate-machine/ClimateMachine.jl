##### Momentum tendencies

export GeostrophicForcing
export Coriolis
export Gravity
export RayleighSponge
using CLIMAParameters.Planet: Omega

#####
##### First order fluxes
#####

function flux(::Momentum, ::Advect, atmos, args)
    @unpack state = args
    return state.ρu .* (state.ρu / state.ρ)'
end
function two_point_flux(
    ::AbstractKennedyGruberSplitForm,
    ::Momentum,
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
    return ρ_ave * u_ave .* u_ave'
end

function flux(::Momentum, ::PressureGradient, atmos, args)
    @unpack state, aux = args
    @unpack ts = args.precomputed
    s = state.ρu * state.ρu'
    pad = SArray{Tuple{size(s)...}}(ntuple(i -> 0, length(s)))
    ref_state = reference_state(atmos)
    if ref_state isa HydrostaticState && ref_state.subtract_off
        return pad + (air_pressure(ts) - aux.ref_state.p) * I
    else
        return pad + air_pressure(ts) * I
    end
end

function two_point_flux(
    ::AbstractKennedyGruberSplitForm,
    ::Momentum,
    ::GravityFluctuation,
    atmos,
    args,
)
    @unpack state1, aux1, state2, aux2 = args
    FT = eltype(state1)

    ρ1 = state1.ρ
    Φ1 = aux1.orientation.Φ

    ρ2 = state2.ρ
    Φ2 = aux2.orientation.Φ

    ref_state = reference_state(atmos)
    if ref_state isa HydrostaticState && ref_state.subtract_off
        ρ1 -= aux1.ref_state.ρ
        ρ2 -= aux2.ref_state.ρ
    end

    ρ_ave = (ρ1 + ρ2) / 2
    Φ_diff = (Φ1 - Φ2) / 2
    return -ρ_ave * Φ_diff * SMatrix{3, 3, FT}(I)
end

#####
##### Second order fluxes
#####

struct ViscousStress <: TendencyDef{Flux{SecondOrder}} end

function flux(::Momentum, ::ViscousStress, atmos, args)
    @unpack state = args
    @unpack τ = args.precomputed.turbulence
    pad = SArray{Tuple{size(τ)...}}(ntuple(i -> 0, length(τ)))
    return pad + τ * state.ρ
end

function flux(::Momentum, ::MoistureDiffusion, atmos, args)
    @unpack state, diffusive = args
    @unpack D_t = args.precomputed.turbulence
    d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
    return d_q_tot .* state.ρu'
end

function flux(::Momentum, ::HyperdiffViscousFlux, atmos, args)
    @unpack state, hyperdiffusive = args
    return state.ρ * hyperdiffusive.hyperdiffusion.ν∇³u_h
end

#####
##### Sources
#####

struct Gravity <: TendencyDef{Source} end

prognostic_vars(::Gravity) = (Momentum(),)

function source(::Momentum, ::Gravity, m, args)
    @unpack state, aux = args
    ref_state = reference_state(m)
    if ref_state isa HydrostaticState && ref_state.subtract_off
        return -(state.ρ - aux.ref_state.ρ) * aux.orientation.∇Φ
    else
        return -state.ρ * aux.orientation.∇Φ
    end
end

struct Coriolis <: TendencyDef{Source} end

prognostic_vars(::Coriolis) = (Momentum(),)

function source(::Momentum, ::Coriolis, m, args)
    @unpack state = args
    FT = eltype(state)
    param_set = parameter_set(m)
    _Omega::FT = Omega(param_set)
    # note: this assumes a SphericalOrientation
    return -SVector(0, 0, 2 * _Omega) × state.ρu
end

struct GeostrophicForcing{FT} <: TendencyDef{Source}
    f_coriolis::FT
    u_geostrophic::FT
    v_geostrophic::FT
end

prognostic_vars(::GeostrophicForcing) = (Momentum(),)

function source(::Momentum, s::GeostrophicForcing, m, args)
    @unpack state, aux = args
    u_geo = SVector(s.u_geostrophic, s.v_geostrophic, 0)
    ẑ = vertical_unit_vector(m, aux)
    fkvector = s.f_coriolis * ẑ
    return -fkvector × (state.ρu .- state.ρ * u_geo)
end

"""
    RayleighSponge{FT} <: TendencyDef{Source}

Rayleigh Damping (Linear Relaxation) for top wall momentum components
Assumes laterally periodic boundary conditions for LES flows. Momentum components
are relaxed to reference values (zero velocities) at the top boundary.
"""
struct RayleighSponge{FT} <: TendencyDef{Source}
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Relaxation velocity components"
    u_relaxation::SVector{3, FT}
    "Sponge exponent"
    γ::FT
end

prognostic_vars(::RayleighSponge) = (Momentum(),)

function source(::Momentum, s::RayleighSponge, m, args)
    @unpack state, aux = args
    z = altitude(m, aux)
    if z >= s.z_sponge
        r = (z - s.z_sponge) / (s.z_max - s.z_sponge)
        β_sponge = s.α_max * sinpi(r / 2)^s.γ
        return -β_sponge * (state.ρu .- state.ρ * s.u_relaxation)
    else
        FT = eltype(state)
        return SVector{3, FT}(0, 0, 0)
    end
end
