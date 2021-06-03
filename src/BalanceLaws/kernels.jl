@inline function total_flux_first_order!(
    bl::BalanceLaw,
    flux::Grad{S},
    state,
    aux,
    t,
    direction,
) where {S}

    flux_first_order!(bl, flux, state, aux, t, direction)

    flux = parent(flux)
    flux2 = similar(flux)
    fill!(flux2, -zero(eltype(flux)))
    two_point_flux_first_order!(bl, Grad{S}(flux2), state, aux, state, aux, t)
    flux .+= flux2
end

"""
    flux_first_order!(
        bl::BalanceLaw,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real
    )

Computes (and assembles) flux terms `F¹(Y)` in:

```
∂Y
-- + ∇ • F¹(Y) + ∇ • F²(Y,G) = S(Y, G),     G = ∇Y
∂t
```

Computes and assembles non-diffusive
fluxes in the model equations.

For this fallback to work, several methods must be defined:
 - [`prognostic_vars`](@ref)
 - [`eq_tends`](@ref)
 - [`get_prog_state`](@ref)
optionally,
 - [`precompute`](@ref)
and individual [`flux`](@ref) kernels that
are defined for each type that `eq_tends` returns.
"""
@inline function flux_first_order!(
    bl::BalanceLaw,
    flux,
    state,
    aux,
    t,
    direction,
)

    tend = Flux{FirstOrder}()
    _args = (; state, aux, t, direction)
    args = merge(_args, (precomputed = precompute(bl, _args, tend),))

    map(prognostic_vars(bl)) do prog
        var, name = get_prog_state(flux, prog)
        val = Σfluxes(prog, eq_tends(prog, bl, tend), bl, args)
        setproperty!(var, name, val)
    end
    nothing
end

@inline function two_point_flux_first_order!(
    bl::BalanceLaw,
    flux,
    state1,
    aux1,
    state2,
    aux2,
    t,
)
    tend = FluxDifferencing{FirstOrder}()
    _args = (; state1, aux1, state2, aux2, t)
    #args = merge(_args, (precomputed = precompute(bl, _args, tend),))
    # TODO: handle precompute
    args = _args

    map(prognostic_vars(bl)) do prog
        var, name = get_prog_state(flux, prog)
        val = Σtwo_point_fluxes(prog, eq_tends(prog, bl, tend), bl, args)
        setproperty!(var, name, val)
    end
    nothing
end

"""
    flux_second_order!(
        bl::BalanceLaw,
        flux::Grad,
        state::Vars,
        diffusive::Vars,
        hyperdiffusive::Vars,
        aux::Vars,
        t::Real
    )

Computes (and assembles) flux terms `F²(Y, G)` in:

```
∂Y
-- + ∇ • F¹(Y) + ∇ • F²(Y,G) = S(Y, G),     G = ∇Y
∂t
```

Diffusive fluxes in BalanceLaw. Viscosity, diffusivity are calculated
in the turbulence subcomponent and accessed within the diffusive flux
function. Contributions from subcomponents are then assembled (pointwise).

For this fallback to work, several methods must be defined:
 - [`prognostic_vars`](@ref)
 - [`eq_tends`](@ref)
 - [`get_prog_state`](@ref)
optionally,
 - [`precompute`](@ref)
and individual [`flux`](@ref) kernels that
are defined for each type that `eq_tends` returns.
"""
@inline function flux_second_order!(
    bl::BalanceLaw,
    flux,
    state,
    diffusive,
    hyperdiffusive,
    aux,
    t,
)
    tend = Flux{SecondOrder}()
    _args = (; state, aux, t, diffusive, hyperdiffusive)
    args = merge(_args, (precomputed = precompute(bl, _args, tend),))

    map(prognostic_vars(bl)) do prog
        var, name = get_prog_state(flux, prog)
        val = Σfluxes(prog, eq_tends(prog, bl, tend), bl, args)
        setproperty!(var, name, val)
    end
    nothing
end

"""
    source!(
        bl::BalanceLaw,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction::Direction,
    )

Computes (and assembles) source terms `S(Y)` in:

```
∂Y
-- + ∇ • F¹(Y) + ∇ • F²(Y,G) = S(Y, G),     G = ∇Y
∂t
```

For this fallback to work, several methods must be defined:
 - [`prognostic_vars`](@ref)
 - [`eq_tends`](@ref)
 - [`get_prog_state`](@ref)
optionally,
 - [`precompute`](@ref)
and individual [`source`](@ref) kernels that
are defined for each type that `eq_tends` returns.
"""
function source!(bl::BalanceLaw, source, state, diffusive, aux, t, direction)
    tend = Source()
    _args = (; state, aux, t, direction, diffusive)
    args = merge(_args, (precomputed = precompute(bl, _args, tend),))

    map(prognostic_vars(bl)) do prog
        var, name = get_prog_state(source, prog)
        val = Σsources(prog, eq_tends(prog, bl, tend), bl, args)
        setproperty!(var, name, val)
    end
    nothing
end
