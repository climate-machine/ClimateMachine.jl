#### Balance Law Interface
"""
    abstract type BalanceLaw end

An abstract type representing a PDE balance law of the form:

```math
\\frac{dq}{dt} = \\nabla \\cdot F_1(q, a, t) + \\nabla \\cdot F_2(q, \\nabla g, h, a, t) + S(q, \\nabla g, a, t)
```
where:
- ``q`` is the prognostic state,
- ``a`` is the auxiliary state,
- ``g = G(q, a, t)`` is the gradient state (variables of which we compute the
  gradient),
- ``h`` is the hyperdiffusive state.

Subtypes of `BalanceLaw` should define the following interfaces:
- [`vars_state`](@ref) to define the prognostic, auxiliary and intermediate
  variables.
- [`flux_first_order!`](@ref) to compute ``F_1``
- [`flux_second_order!`](@ref) to compute ``F_2``
- [`source!`](@ref) to compute ``S``

If `vars(bl, ::GradientFlux, FT)` is non-empty, then the following should be
defined:
- [`compute_gradient_argument!`](@ref) to compute ``G``
- [`compute_gradient_flux!`](@ref) is a linear transformation of ``\\nabla g``

If `vars(bl, ::Hyperdiffusive, FT)` is non-empty, then the following should be
defined:
- [`transform_post_gradient_laplacian!`](@ref)

Additional functions:
- [`wavespeed`](@ref) if using the Rusanov numerical flux.
- [`boundary_state!`](@ref) if using non-periodic boundary conditions.
"""
abstract type BalanceLaw end

"""
    BalanceLaws.vars_state(::BL, ::AbstractStateType, FT)

Defines the state variables of a [`BalanceLaw`](@ref) subtype `BL` with floating
point type `FT`.

For each [`AbstractStateType`](@ref), this should return a `NamedTuple` type,
with element type either `FT`, an `SArray` with element type `FT` or another
`NamedTuple` satisfying the same property.

For convenience, we recommend using the [`VariableTemplates.@vars`](@ref) macro.

# Example
```julia
struct MyBalanceLaw <: BalanceLaw end

BalanceLaws.vars_state(::MyBalanceLaw, ::Prognostic, FT) =
    @vars(x::FT, y::SVector{3, FT})
BalanceLaws.vars_state(::MyBalanceLaw, ::Auxiliary, FT) =
    @vars(components::@vars(a::FT, b::FT))
```
"""
function vars_state end

# Fallback: no variables
vars_state(::BalanceLaw, ::AbstractStateType, FT) = @vars()

"""
    init_state_prognostic!(
        ::BL,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        localgeo,
        args...,
    )

Sets the initial state of the prognostic variables `state_prognostic` at each
node for a [`BalanceLaw`](@ref) subtype `BL`.
"""
function init_state_prognostic! end

# TODO: make these functions consistent with init_state_prognostic!
"""
    nodal_init_state_auxiliary!(::BL, state_auxiliary, state_temporary, geom)

Sets the initial state of the auxiliary variables `state_auxiliary` at each
node for a [`BalanceLaw`](@ref) subtype `BL`.

See also [`init_state_auxiliary!`](@ref).
"""
function nodal_init_state_auxiliary!(m::BalanceLaw, aux, tmp, geom) end


"""
    init_state_auxiliary!(
        ::BL,
        statearray_auxiliary,
        geom::LocalGeometry,
    )

Sets the initial state of the auxiliary variables `state_auxiliary` at each node
for a [`BalanceLaw`](@ref) subtype `BL`. By default this calls
[`nodal_init_state_auxiliary!`](@ref).
"""
function init_state_auxiliary!(
    balance_law::BalanceLaw,
    statearray_auxiliary,
    grid,
    direction,
)
    init_state_auxiliary!(
        balance_law,
        nodal_init_state_auxiliary!,
        statearray_auxiliary,
        grid,
        direction,
    )
end

"""
    flux_first_order!(
        ::BL,
        flux::Grad,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        t::Real,
        direction
    )

Sets the first-order (hyperbolic) `flux` terms for a [`BalanceLaw`](@ref) subtype `BL`.
"""
function flux_first_order! end
function two_point_flux_first_order! end

"""
    flux_second_order!(
        ::BL,
        flux::Grad,
        state_prognostic::Vars,
        state_gradient_flux::Vars,
        hyperdiffusive::Vars,
        state_auxiliary::Vars,
        t::Real
    )

Sets second-order (parabolic) `flux` terms for a [`BalanceLaw`](@ref) subtype `BL`.
"""
function flux_second_order! end

"""
    source!(
        ::BL,
        source::Vars,
        state_prognostic::Vars,
        diffusive::Vars,
        state_auxiliary::Vars,
        t::Real
    )

Compute non-conservative source terms for a [`BalanceLaw`](@ref) subtype `BL`.
"""
function source! end

"""
    compute_gradient_argument!(
        ::BL,
        transformstate::Vars,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        t::Real
    )

Transformation of state variables `state_prognostic` to variables
`transformstate` of which gradients are computed for a [`BalanceLaw`](@ref)
subtype `BL`.
"""
function compute_gradient_argument! end

function compute_gradient_argument!(
    balance_law::BalanceLaw,
    transformstate::AbstractArray,
    state_prognostic::AbstractArray,
    state_auxiliary::AbstractArray,
    t,
)
    FT = eltype(transformstate)
    compute_gradient_argument!(
        balance_law,
        Vars{vars_state(balance_law, Gradient(), FT)}(transformstate),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary),
        t,
    )
end


"""
    compute_gradient_flux!(
        ::BL,
        state_gradient_flux::Vars,
        ∇transformstate::Grad,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        t::Real
    )

Transformation of gradients to the diffusive variables for a
[`BalanceLaw`](@ref) subtype `BL`. This should be a linear function of
`∇transformstate`
"""
function compute_gradient_flux! end

function compute_gradient_flux!(
    balance_law::BalanceLaw,
    state_gradient_flux::AbstractArray,
    ∇transformstate::AbstractArray,
    state_prognostic::AbstractArray,
    state_auxiliary::AbstractArray,
    t,
)
    FT = eltype(state_gradient_flux)
    compute_gradient_flux!(
        balance_law,
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux),
        Grad{vars_state(balance_law, Gradient(), FT)}(∇transformstate),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary),
        t,
    )
end

"""
    transform_post_gradient_laplacian!(
        ::BL,
        Qhypervisc_div::Vars,
        ∇Δtransformstate::Grad,
        state_auxiliary::Vars,
        t::Real
    )

Transformation of Laplacian gradients to the hyperdiffusive variables for a
[`BalanceLaw`](@ref) subtype `BL`.
"""
function transform_post_gradient_laplacian! end

"""
    wavespeed(
        ::BL,
        n⁻,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        t::Real,
        direction
    )

Wavespeed in the direction `n⁻` for a [`BalanceLaw`](@ref) subtype `BL`. This is
required to be defined if using a `RusanovNumericalFlux` numerical flux.
"""
function wavespeed end


"""
    update_auxiliary_state!(
        dg::DGModel,
        m::BalanceLaw,
        statearray_aux,
        t::Real,
        elems::UnitRange,
        [diffusive=false]
    )

Hook to update the auxiliary state variables before calling any other functions.

By default, this calls [`nodal_update_auxiliary_state!`](@ref) at each node.

If `diffusive=true`, then `state_gradflux` is also passed to
`nodal_update_auxiliary_state!`.
"""
function update_auxiliary_state!(
    dg,
    balance_law::BalanceLaw,
    state_prognostic,
    t,
    elems,
    diffusive = false,
)
    update_auxiliary_state!(
        nodal_update_auxiliary_state!,
        dg,
        balance_law,
        state_prognostic,
        t,
        elems;
        diffusive = diffusive,
    )
end

"""
    nodal_update_auxiliary_state!(::BL, state_prognostic, state_auxiliary, [state_gradflux,] t)

Update the auxiliary state variables at each node for a [`BalanceLaw`](@ref)
subtype `BL`. By default it does nothing.

Called by [`update_auxiliary_state!`](@ref).
"""
function nodal_update_auxiliary_state!(args...)
    nothing
end

"""
    update_auxiliary_state_gradient!(
        dg::DGModel,
        m::BalanceLaw,
        statearray_aux,
        t::Real,
        elems::UnitRange,
        [diffusive=false]
    )

Hook to update the auxiliary state variables after the gradient computation.

By default, this calls nothing.

If `diffusive=true`, then `state_gradflux` is also passed to
`nodal_update_auxiliary_state!`.
"""
function update_auxiliary_state_gradient! end

# upward integrals
"""
    integral_load_auxiliary_state!(::BL, integrand, state_prognostic, state_aux)

Specify variables `integrand` which will have their upward integrals computed.

See also [`UpwardIntegrals`](@ref)
"""
function integral_load_auxiliary_state! end

"""
    integral_set_auxiliary_state!(::BL, state_aux, integral)

Update auxiliary variables based on the upward integral `integral` defined in
[`integral_load_auxiliary_state!`](@ref).
"""
function integral_set_auxiliary_state! end

"""
    indefinite_stack_integral!

Compute indefinite integral along stack.
"""
function indefinite_stack_integral! end

# downward integrals
"""
    reverse_integral_load_auxiliary_state!(::BL, integrand, state_prognostic, state_aux)

Specify variables `integrand` which will have their downward integrals computed.

See also [`DownwardIntegrals`](@ref)
"""
function reverse_integral_load_auxiliary_state! end

"""
    reverse_integral_set_auxiliary_state!(::BL, state_aux, integral)

Update auxiliary variables based on the downward integral `integral` defined in
[`reverse_integral_load_auxiliary_state!`](@ref).
"""
function reverse_integral_set_auxiliary_state! end

"""
    reverse_indefinite_stack_integral!

Compute reverse indefinite integral along stack.
"""
function reverse_indefinite_stack_integral! end

"""
    function state_to_entropy_variables!(
        balancelaw::BalanceLaw,
        state_entropy::Vars,
        state_prognostic::Vars,
        state_auxiliary::Vars,
    )
Compute entropy variables `state_entropy` from a given `state_prognostic` and
`state_auxiliary`
"""
function state_to_entropy_variables!(
    balancelaw::BalanceLaw,
    entropy::AbstractArray{FT},
    state::AbstractArray{FT},
    aux::AbstractArray{FT},
) where {FT}
    state_to_entropy_variables!(
        balancelaw,
        Vars{vars_state(balancelaw, Entropy(), FT)}(entropy),
        Vars{vars_state(balancelaw, Prognostic(), FT)}(state),
        Vars{vars_state(balancelaw, Auxiliary(), FT)}(aux),
    )
end

"""
    function entropy_variables_to_state!(
        balancelaw::BalanceLaw,
        state_prognostic::Vars,
        state_auxiliary::Vars,
        state_entropy::Vars,
    )
Compute `state_prognostic` and relevant componennts of `state_auxiliary` from
the entropy variables `state_entropy`
"""
function entropy_variables_to_state!(
    balancelaw::BalanceLaw,
    state::AbstractArray{FT},
    aux::AbstractArray{FT},
    entropy::AbstractArray{FT},
) where {FT}
    entropy_variables_to_state!(
        balancelaw,
        Vars{vars_state(balancelaw, Prognostic(), FT)}(state),
        Vars{vars_state(balancelaw, Auxiliary(), FT)}(aux),
        Vars{vars_state(balancelaw, Entropy(), FT)}(entropy),
    )
end

"""
    function state_to_entropy(
        balancelaw::BalanceLaw,
        state_prognostic::Vars,
        state_auxiliary::Vars,
    )
Compute entropy from a given `state_prognostic` and `state_auxiliary`
"""
function state_to_entropy(
    balancelaw::BalanceLaw,
    state::AbstractArray{FT},
    aux::AbstractArray{FT},
) where {FT}
    state_to_entropy(
        balancelaw,
        Vars{vars_state(balancelaw, Prognostic(), FT)}(state),
        Vars{vars_state(balancelaw, Auxiliary(), FT)}(aux),
    )
end



# Internal methods
number_states(m::BalanceLaw, st::AbstractStateType, FT = Int) =
    varsize(vars_state(m, st, FT))

### split explicit functions
function initialize_states! end
function tendency_from_slow_to_fast! end
function cummulate_fast_solution! end
function reconcile_from_fast_to_slow! end

parameter_set(balance_law) = balance_law.param_set

"""
    sub_model(::BalanceLaw, ::Type{AbstractSubModel})

Returns a tuple of balance law properties
whose supertypes are `AbstractSubModel`.
"""
function sub_model end
