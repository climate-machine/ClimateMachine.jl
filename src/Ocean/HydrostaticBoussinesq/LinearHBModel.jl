export LinearHBModel

# Linear model for 1D IMEX
"""
    LinearHBModel <: BalanceLaw

A `BalanceLaw` for modeling vertical diffusion implicitly.

write out the equations here

# Usage

    model = HydrostaticBoussinesqModel(problem)
    linear = LinearHBModel(model)

"""
struct LinearHBModel{M} <: BalanceLaw
    ocean::M
    function LinearHBModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end

"""
    Copy over state, aux, and diff variables from HBModel
"""
vars_state_conservative(lm::LinearHBModel, FT) =
    vars_state_conservative(lm.ocean, FT)
vars_state_gradient(lm::LinearHBModel, FT) = vars_state_gradient(lm.ocean, FT)
vars_state_gradient_flux(lm::LinearHBModel, FT) =
    vars_state_gradient_flux(lm.ocean, FT)
vars_state_auxiliary(lm::LinearHBModel, FT) = vars_state_auxiliary(lm.ocean, FT)
vars_integrals(lm::LinearHBModel, FT) = @vars()

"""
    No integration, hyperbolic flux, or source terms
"""
@inline integrate_aux!(::LinearHBModel, _...) = nothing
@inline flux_first_order!(::LinearHBModel, _...) = nothing
@inline source!(::LinearHBModel, _...) = nothing

"""
    No need to init, initialize by full model
"""
init_state_auxiliary!(lm::LinearHBModel, _...) = nothing
init_state_conservative!(lm::LinearHBModel, _...) = nothing

"""
    Copy over second-order term computation from ocean model
"""
compute_gradient_argument!(lm::LinearHBModel, args...) =
    compute_gradient_argument!(lm.ocean, args...)
compute_gradient_flux!(lm::LinearHBModel, args...) =
    compute_gradient_argument!(lm.ocean, args...)
flux_second_order!(lm::LinearHBModel, args...) =
    flux_second_order!(lm.ocean, args...)
wavespeed(lm::LinearHBModel, args...) = wavespeed(lm.ocean, args...)
