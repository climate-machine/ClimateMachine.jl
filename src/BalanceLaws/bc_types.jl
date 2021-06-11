##### Boundary condition types

import Base
using DispatchedTuples

export DefaultBC, DefaultBCValue, DefaultBCFlux
export set_boundary_values!
export set_boundary_fluxes!
export used_bcs

# The following BC naming convention is used:

# `all_bcs`        - all boundary conditions for a given balance
#                    law (on all faces of the geometry) This is
#                    typically a tuple of individual BCs.
#
# `bcs`            - boundary conditions per variable (a DispatchedTuple)
#                    for a single point in space, or wall, for all variables.
#
# `bc_set_driver`  - a unique `Set` (in the mathematical sense)
#                    of boundary conditions, specified at the
#                    driver, for a single point in space, or wall,
#                    for all variables.
#
# `bc`             - a single boundary condition for a single point
#                    in space, or wall, for a single variable.

# Additional suffixes are used to indicate one of several things:
# - `driver` - BCs described in the driver
# - `default` - balance law-specific BCs

"""
    DefaultBCValue

Default BC value, which results in the default
boundary condition behavior.
"""
struct DefaultBCValue end

"""
    DefaultBCFlux

Default BC flux, which results in the default
boundary condition behavior.
"""
struct DefaultBCFlux end

"""
    DefaultBC

The default boundary condition definition,
which results in yielding the default boundary
condition value [`DefaultBCValue`](@ref).
"""
struct DefaultBC end

# If DefaultBCValue's are mixed with real values,
# then let DefaultBCValue's be zero:
# Base.:+(::DefaultBCValue, x::FT) where {FT <: AbstractFloat} = x
# Base.:+(x::FT, ::DefaultBCValue) where {FT <: AbstractFloat} = x
# Base.:+(::DefaultBCValue, x::SArray{Tuple{N}, FT, 1, N}) where {N, FT} = x
# Base.:+(x::SArray{Tuple{N}, FT, 1, N}, ::DefaultBCValue) where {N, FT} = x

# Base.:+(::DefaultBCFlux, x::FT) where {FT <: AbstractFloat} = x
# Base.:+(x::FT, ::DefaultBCFlux) where {FT <: AbstractFloat} = x
# Base.:+(::DefaultBCFlux, x::SArray{Tuple{N}, FT, 1, N}) where {N, FT} = x
# Base.:+(x::SArray{Tuple{N}, FT, 1, N}, ::DefaultBCFlux) where {N, FT} = x

Base.:+(::DefaultBCValue, x) = x
Base.:+(x, ::DefaultBCValue) = x
Base.:+(::DefaultBCFlux, x) = x
Base.:+(x, ::DefaultBCFlux) = x


"""
    boundary_value(::AbstractPrognosticVariable, bc, bl, args, nf)

Return the value of the boundary condition, given
 - `::AbstractPrognosticVariable` the prognostic variable
 - `bc` the individual boundary condition type
 - `bl` the balance law
 - `args` top-level arguments, packed into a NamedTuple
 - `nf` the numerical flux
"""
boundary_value(::AbstractPrognosticVariable, ::DefaultBC, bl, args, nf) =
    DefaultBCValue()

"""
    boundary_flux(::AbstractPrognosticVariable, bc, bl, args, nf)

Return the prescribed boundary flux, given
 - `::AbstractPrognosticVariable` the prognostic variable
 - `bc` the individual boundary condition type
 - `bl` the balance law
 - `args` top-level arguments, packed into a NamedTuple
 - `nf` the numerical flux
"""
boundary_flux(::AbstractPrognosticVariable, ::DefaultBC, bl, args, nf) =
    DefaultBCFlux()

"""
    default_bcs(::AbstractPrognosticVariable)

A tuple of default boundary condition definitions
for a given prognostic variable.
"""
function default_bcs end

"""
    set_boundary_values!(
        state⁺,
        bl,
        nf,
        bcs,
        args,
        prog_vars = prognostic_vars(bl)
    )

A convenience method for setting `state⁺` inside `boundary_state!`.

Arguments:
 - `state⁺` the exterior state
 - `bl` the balance law
 - `bcs` the balance law boundary conditions on a given boundary
 - `nf::Union{NumericalFluxFirstOrder,NumericalFluxGradient}` the numerical flux
 - `args` the top-level arguments
 - `prog_vars` (optional) the balance law's prognostic variables
"""
function set_boundary_values!(
    state⁺,
    bl,
    nf,
    bcs,
    args,
    prog_vars = prognostic_vars(bl),
)
    state⁻ = args.state⁻
    bcs_used = used_bcs(bcs)
    map(prog_vars) do pv
        var⁺, name = get_prog_state(state⁺, pv)
        var⁻, name = get_prog_state(state⁻, pv)
        bcvals = map(bcs_used[pv]) do bc
            boundary_value(pv, bc, bl, args, nf)
        end
        set_bc!(var⁺, name, bcvals)
    end
end

"""
    set_boundary_fluxes!(
        fluxᵀn,
        bl,
        nf,
        bcs,
        args,
        prog_vars = prognostic_vars(bl)
    )

A convenience method for setting the
numerical (boundary) flux `fluxᵀn` in
`normal_boundary_flux_second_order!`.

Arguments:
 - `fluxᵀn` the numerical (boundary) flux
 - `bl` the balance law
 - `bcs` the balance law boundary conditions on a given boundary
 - `nf::NumericalFluxSecondOrder` the numerical flux
 - `args` the top-level arguments
 - `prog_vars` (optional) the balance law's prognostic variables
"""
function set_boundary_fluxes!(
    fluxᵀn,
    bl,
    nf,
    bcs,
    args,
    prog_vars = prognostic_vars(bl),
)
    bcs_used = used_bcs(bcs)
    map(prog_vars) do pv
        varflux, name = get_prog_state(fluxᵀn, pv)
        bfluxes = map(bcs_used[pv]) do bc
            boundary_flux(pv, bc, bl, args, nf)
        end
        Σbfluxes = sum_boundary_fluxes(bfluxes)
        varprop = getproperty(varflux, name)
        varflux_assigned = Σbfluxes + varprop
        setproperty!(varflux, name, varflux_assigned)
    end
end

#####
##### Internal methods for setting/diagonalizing boundary conditions
#####

sum_boundary_fluxes(bcvals::NTuple{N, DefaultBCFlux}) where {N} =
    DefaultBCFlux()
sum_boundary_fluxes(bcvals) = sum(bcvals)

# Internal method: call default_bcs for bl-specific
# prognostic variables:
function default_bcs(bl)
    tup = map(prognostic_vars(bl)) do pv
        map(default_bcs(pv)) do bc
            @assert pv isa AbstractPrognosticVariable
            (pv, bc)
        end
    end
    tup = tuple_of_tuples(tup)
    return DispatchedTuple(tup, DefaultBC())
end

"""
    used_bcs(::BoundaryCondition)

Return the used Boundary Conditions (BCs)
given the `BoundaryCondition`.

BCs are prescribed with 3 levels of precedence:

1) Driver/experiment (highest precedence)
2) Balance-law specific (medium precedence)
3) Balance-law generic (lowest precedence)

`used_bcs` returns the bundle of BCs from all
three categories above and takes the ones with
the highest precedence per variable.
"""
function used_bcs end

function used_bcs(pv::AbstractPrognosticVariable, bcs, bc_default)
    bc_driver = bcs[pv]
    if bc_driver == (DefaultBC(),) # use bl-specific BCs:
        bc_used = bc_default[pv]
    else # use driver-prescribed BCs:
        bc_used = bc_driver
    end
    return bc_used
end
function used_bcs(bl, bc_set_driver)
    bcs_driver = map(bc_set_driver) do bc
        map(prognostic_vars(bc)) do pv
            (pv, bc)
        end
    end
    bcs_driver = DispatchedTuple(tuple_of_tuples(bcs_driver), DefaultBC())
    bc_default = default_bcs(bl)
    bcs_used = DispatchedSet(map(prognostic_vars(bl)) do pv
        (pv, used_bcs(pv, bcs_driver, bc_default))
    end)
    return bcs_used
end

# TODO: remove Tuple{DefaultBC} method:
# TODO: why is NTuple{N, DefaultBC}) where {N} needed?
set_bc!(var⁺, name, bcvals::Tuple{DefaultBCValue}) = nothing
set_bc!(var⁺, name, bcvals::NTuple{N, DefaultBCValue}) where {N} = nothing
set_bc!(var⁺, name, bcvals) = setproperty!(var⁺, name, sum(bcvals))

"""
    bc_precompute(bc, bl, args, dispatch_helper)

A nested NamedTuple of precomputed (cached) values
and or objects. This is useful for "expensive"
point-wise quantities that are used in multiple
boundary condition functions. For example, computing
a quantity that requires iteration.

This is a separated from [`precompute`](@ref), as
simply because there are many `precompute` definitions,
and splitting the methods helps search-ability.
"""
bc_precompute(bc, bl, args, dispatch_helper) = NamedTuple()
