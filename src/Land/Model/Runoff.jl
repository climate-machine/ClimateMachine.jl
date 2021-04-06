module Runoff

using LinearAlgebra
using DocStringExtensions

using Printf
using ...VariableTemplates
using ...Land: SoilModel, pressure_head, hydraulic_conductivity, get_temperature

export AbstractPrecipModel,
    DrivenConstantPrecip,
    AbstractSurfaceRunoffModel,
    NoRunoff,
    compute_surface_grad_bc,
    CoarseGridRunoff

"""
    AbstractPrecipModel{FT <: AbstractFloat}
"""
abstract type AbstractPrecipModel{FT <: AbstractFloat} end

"""
    DrivenConstantPrecip{FT, F} <: AbstractPrecipModel{FT}

Instance of a precipitation distribution where the precipication value
is constant across the domain. However, this value can change in time.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct DrivenConstantPrecip{FT, F} <: AbstractPrecipModel{FT}
    "Mean precipitation in grid"
    mp::F
    function DrivenConstantPrecip{FT}(mp::F) where {FT, F}
        new{FT, F}(mp)
    end
end

function (dcp::DrivenConstantPrecip{FT})(t::Real) where {FT}
    return FT(dcp.mp(t))
end

"""
    AbstractSurfaceRunoffModel

Abstract type for different surface runoff models. Currently, only
`NoRunoff` is supported.
"""
abstract type AbstractSurfaceRunoffModel end

"""
    NoRunoff <: AbstractSurfaceRunoffModel

Chosen when no runoff is to be modeled.
"""
struct NoRunoff <: AbstractSurfaceRunoffModel end


"""
    CoarseGridRunoff{FT} <: AbstractSurfaceRunoffModel

Chosen when no subgrid effects are to be modeled.
"""
struct CoarseGridRunoff{FT} <: AbstractSurfaceRunoffModel
    "Mean vertical resolution at the surface"
    Δz::FT
end

"""
    function compute_surface_grad_bc(soil::SoilModel,
                                     runoff_model::CoarseGridRunoff,
                                     precip_model::AbstractPrecipModel,
                                     state⁻::Vars,
                                     diff⁻::Vars,
                                     aux⁻::Vars,
                                     t::Real
                                     )

Given a runoff model and a precipitation distribution function, compute 
the surface water Neumann BC. This can be a function of time, and state.
"""
function compute_surface_grad_bc(
    soil::SoilModel,
    runoff_model::CoarseGridRunoff,
    precip_model::AbstractPrecipModel,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t::Real,
)
    FT = eltype(state⁻)
    incident_water_flux = precip_model(t)
    Δz = runoff_model.Δz
    water = soil.water
    param_functions = soil.param_functions
    hydraulics = water.hydraulics
    ν = param_functions.porosity
    #specific_storage = param_functions.S_s

    T = get_temperature(soil.heat, aux⁻, t)
    θ_i = state⁻.soil.water.θ_i
    # Ponding Dirichlet BC
    ## ONLY for overland flow
    ϑ_bc = ν#+specific_storage*state⁻.river.area#FT(ν - θ_i)
    # Value below surface
    ϑ_below = state⁻.soil.water.ϑ_l

    # Approximate derivative of hydraulic head with respect to z
    ∂h∂z =
        FT(1) +
        (
            pressure_head(hydraulics, param_functions, ϑ_bc, θ_i) -
            pressure_head(hydraulics, param_functions, ϑ_below, θ_i)
        ) / Δz

    K =
        soil.param_functions.Ksat * hydraulic_conductivity(
            water.impedance_factor,
            water.viscosity_factor,
            water.moisture_factor,
            hydraulics,
            θ_i,
            param_functions.porosity,
            T,
            ϑ_bc / ν,# when ice is present, K still measured with ν, not νeff.
        )

    i_c = (K * ∂h∂z)
    if incident_water_flux < -i_c#-norm(diff⁻.soil.water.K∇h) #-i_c#More negative if both are negative,
        #ponding BC
        K∇h⁺ = i_c#min(i_c,-incident_water_flux) #i_c
    else
        #K∇h⁺ = n̂ * (-FT(2) * incident_water_flux) - diff⁻.soil.water.K∇h
        K∇h⁺ = - incident_water_flux
    end
    #@printf("%lf %lf %lf %lf %le %le %le \n", t, aux⁻.x, aux⁻.y, aux⁻.z, -i_c, incident_water_flux, -norm(diff⁻.soil.water.K∇h))
    return K∇h⁺ # now a scalar
end



"""
    function compute_surface_grad_bc(soil::SoilModel,
                                     runoff_model::NoRunoff,
                                     precip_model::AbstractPrecipModel,
                                     state⁻::Vars,
                                     aux⁻::Vars,
                                     diff⁻::Vars,
                                     t::Real
                                     )

Given a runoff model and a precipitation distribution function, compute 
the surface water Neumann BC. This can be a function of time, and state.
"""
function compute_surface_grad_bc(
    soil::SoilModel,
    runoff_model::NoRunoff,
    precip_model::AbstractPrecipModel,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    t::Real,
)
    FT = eltype(state⁻)
    incident_water_flux = precip_model(t)
    #K∇h⁺ = n̂ * (-FT(2) * incident_water_flux) - diff⁻.soil.water.K∇h
    K∇h⁺ = -incident_water_flux
    return K∇h⁺
end

end
