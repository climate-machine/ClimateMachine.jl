module NumericalFluxes

export NumericalFluxGradient,
    NumericalFluxFirstOrder,
    NumericalFluxSecondOrder,
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxFirstOrder,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxHigherOrder


using StaticArrays, LinearAlgebra
using CLIMA.VariableTemplates
using KernelAbstractions.Extras: @unroll
import ..DGmethods:
    BalanceLaw,
    Grad,
    Vars,
    vars_state_conservative,
    vars_state_gradient_flux,
    vars_state_auxiliary,
    vars_state_gradient,
    boundary_state!,
    wavespeed,
    flux_first_order!,
    flux_second_order!,
    compute_gradient_flux!,
    number_state_conservative,
    number_state_gradient,
    compute_gradient_argument!,
    num_gradient_laplacian,
    vars_gradient_laplacian,
    vars_hyperdiffusive,
    transform_post_gradient_laplacian!

"""
    NumericalFluxGradient

Any `P <: NumericalFluxGradient` should define methods for:

   numerical_flux_gradient!(gnf::P, balance_law::BalanceLaw, diffF, n⁻, Q⁻, Qstate_gradient_flux⁻, Qaux⁻, Q⁺,
                            Qstate_gradient_flux⁺, Qaux⁺, t)
   numerical_boundary_flux_gradient!(gnf::P, balance_law::BalanceLaw, local_state_gradient_flux, n⁻, local_transform⁻, local_state_conservative⁻,
                                     local_state_auxiliary⁻, local_transform⁺, local_state_conservative⁺, local_state_auxiliary⁺, bctype, t)

"""
abstract type NumericalFluxGradient end

"""
    CentralNumericalFluxGradient <: NumericalFluxGradient

"""
struct CentralNumericalFluxGradient <: NumericalFluxGradient end

function numerical_flux_gradient!(
    ::CentralNumericalFluxGradient,
    balance_law::BalanceLaw,
    states::NamedTuple,
    t,
) where {T, S, A}

    states.gradient .=
        normal_vector .*
        (parent(states.argument⁺) .+ parent(states.argument⁻))' ./ 2
end

function numerical_boundary_flux_gradient!(
    numerical_flux::CentralNumericalFluxGradient,
    balance_law::BalanceLaw,
    states::NamedTuple,
    normal_vector::SVector,
    bctype,
    t,
    states1⁻::NamedTuple,
) where {D, T, S, A}
    states_boundary = (
        conservative⁻ = states.conservative⁻,
        auxiliary⁻ = states.auxiliary⁻,
        conservative⁺ = states.conservative⁺,
        auxiliary⁺ = states.auxiliary⁺,
    )

    boundary_state!(
        numerical_flux,
        balance_law,
        states_boundary,
        normal_vector,
        bctype,
        t,
        states1⁻,
    )

    states_argument = (
        argument = states.argument⁺,
        conservative = states.conservative⁺,
        auxiliary = states.auxiliary⁺,
    )

    compute_gradient_argument!(balance_law, states_argument, t)
    states.gradient .= normal_vector .* parent(states.argument⁺)'
end

"""
    NumericalFluxFirstOrder

Any `N <: NumericalFluxFirstOrder` should define the a method for

    numerical_flux_first_order!(numerical_flux::N, balance_law::BalanceLaw, flux, normal_vector⁻, Q⁻, Qaux⁻, Q⁺,
                                 Qaux⁺, t)

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_first_order!(numerical_flux::N, balance_law::BalanceLaw, flux, normal_vector⁻, Q⁻,
                                          Qaux⁻, Q⁺, Qaux⁺, bctype, t)

"""
abstract type NumericalFluxFirstOrder end

function numerical_flux_first_order! end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    balance_law::BalanceLaw,
    states::NamedTuple,
    normal_vector::SVector,
    bctype,
    t,
    states1⁻::NamedTuple,
) where {S, A}

    boundary_state!(
        numerical_flux,
        balance_law,
        states,
        normal_vector,
        bctype,
        t,
        states1⁻,
    )

    numerical_flux_first_order!(
        numerical_flux,
        balance_law,
        states,
        normal_vector,
        t,
    )
end


"""
    RusanovNumericalFlux <: NumericalFluxFirstOrder

The RusanovNumericalFlux (aka local Lax-Friedrichs) numerical flux.

# Usage

    RusanovNumericalFlux()

Requires a `flux_first_order!` and `wavespeed` method for the balance law.
"""
struct RusanovNumericalFlux <: NumericalFluxFirstOrder end

update_penalty!(::RusanovNumericalFlux, ::BalanceLaw, _...) = nothing

function numerical_flux_first_order!(
    numerical_flux::RusanovNumericalFlux,
    balance_law::BalanceLaw,
    states::NamedTuple,
    normal_vector::SVector,
    t,
) where {S, A}

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        states,
        normal_vector,
        t,
    )

    states⁻ =
        (conservative = states.conservative⁻, auxiliary = states.auxiliary⁻)
    states⁺ =
        (conservative = states.conservative⁺, auxiliary = states.auxiliary⁺)

    wavespeed⁻ = wavespeed(balance_law, states⁻, normal_vector, t)
    wavespeed⁺ = wavespeed(balance_law, states⁺, normal_vector, t)

    max_wavespeed = max(wavespeed⁻, wavespeed⁺)
    penalty =
        max_wavespeed *
        (parent(states.conservative⁻) - parent(states.conservative⁺))

    states_less_flux = (
        conservative⁻ = states.conservative⁻,
        auxiliary⁻ = states.auxiliary⁻,
        conservative⁺ = states.conservative⁺,
        auxiliary⁺ = states.auxiliary⁺,
    )

    # TODO: should this operate on ΔQ or penalty?
    update_penalty!(
        numerical_flux,
        balance_law,
        normal_vector,
        max_wavespeed,
        Vars{S}(penalty),
        states_less_flux,
        t,
    )

    fluxᵀn = parent(states.flux)
    fluxᵀn .+= penalty / 2
end

"""
    CentralNumericalFluxFirstOrder() <: NumericalFluxFirstOrder

The central numerical flux for nondiffusive terms

# Usage

    CentralNumericalFluxFirstOrder()

Requires a `flux_first_order!` method for the balance law.
"""
struct CentralNumericalFluxFirstOrder <: NumericalFluxFirstOrder end

function numerical_flux_first_order!(
    ::CentralNumericalFluxFirstOrder,
    balance_law::BalanceLaw,
    states::NamedTuple,
    normal_vector::SVector,
    t,
) where {S, A}

    FT = eltype(states.flux)
    num_state_conservative = number_state_conservative(balance_law, FT)

    flux⁻ = similar(states.flux, Size(3, num_state_conservative))
    fill!(flux⁻, -zero(FT))
    states⁻ = (
        flux = flux⁻,
        conservative = states.conservative⁻,
        auxiliary = states.auxiliary⁻,
    )
    flux_first_order!(balance_law, states⁻, t)

    flux⁺ = similar(states.flux, Size(3, num_state_conservative))
    fill!(flux⁺, -zero(FT))
    states⁺ = (
        flux = flux⁺,
        conservative = states.conservative⁺,
        auxiliary = states.auxiliary⁺,
    )
    flux_first_order!(balance_law, states⁺, t)

    fluxᵀn = parent(states.flux)
    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector / 2)
end

"""
    NumericalFluxSecondOrder

Any `N <: NumericalFluxSecondOrder` should define the a method for

    numerical_flux_second_order!(numerical_flux::N, balance_law::BalanceLaw, flux, normal_vector⁻, Q⁻, Qstate_gradient_flux⁻, Qaux⁻, Q⁺,
                              Qstate_gradient_flux⁺, Qaux⁺, t)

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive diffusive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_second_order!(numerical_flux::N, balance_law::BalanceLaw, flux, normal_vector⁻, Q⁻, Qstate_gradient_flux⁻,
                                       Qaux⁻, Q⁺, Qstate_gradient_flux⁺, Qaux⁺, bctype, t)

"""
abstract type NumericalFluxSecondOrder end

function numerical_flux_second_order! end

function numerical_boundary_flux_second_order! end

"""
    CentralNumericalFluxSecondOrder <: NumericalFluxSecondOrder

The central numerical flux for diffusive terms

# Usage

    CentralNumericalFluxSecondOrder()

Requires a `flux_second_order!` for the balance law.
"""
struct CentralNumericalFluxSecondOrder <: NumericalFluxSecondOrder end

function numerical_flux_second_order!(
    ::CentralNumericalFluxSecondOrder,
    balance_law::BalanceLaw,
    states::NamedTuple,
    normal_vector⁻::SVector,
    t,
) where {S, D, HD, A}

    FT = eltype(states.flux)
    num_state_conservative = number_state_conservative(balance_law, FT)

    flux⁻ = similar(states.flux, Size(3, num_state_conservative))
    fill!(flux⁻, -zero(FT))
    states⁻ = (
        flux = flux⁻,
        conservative = states.conservative⁻,
        auxiliary = states.auxiliary⁻,
        gradient_flux = states.gradient_flux⁻,
        hyperdiffusive = states.hyperdiffusive⁻,
    )
    flux_second_order!(balance_law, states⁻, t)

    flux⁺ = similar(states.flux, Size(3, num_state_conservative))
    fill!(flux⁺, -zero(FT))
    states⁺ = (
        flux = flux⁺,
        conservative = states.conservative⁺,
        auxiliary = states.auxiliary⁺,
        gradient_flux = states.gradient_flux⁺,
        hyperdiffusive = states.hyperdiffusive⁺,
    )
    flux_second_order!(balance_law, states⁺, t)

    fluxᵀn = parent(states.flux)
    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector⁻ / 2)
end

abstract type DivNumericalPenalty end
struct CentralNumericalFluxDivergence <: DivNumericalPenalty end

function numerical_flux_divergence!(
    ::CentralNumericalFluxDivergence,
    balance_law::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
) where {GL}
    parent(div_penalty) .=
        (parent(grad⁺) .- parent(grad⁻))' * (normal_vector / 2)
end

function numerical_boundary_flux_divergence!(
    numerical_flux::CentralNumericalFluxDivergence,
    balance_law::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
    bctype,
) where {GL}
    boundary_state!(
        numerical_flux,
        balance_law,
        grad⁺,
        normal_vector,
        grad⁻,
        bctype,
    )
    numerical_flux_divergence!(
        numerical_flux,
        balance_law,
        div_penalty,
        normal_vector,
        grad⁻,
        grad⁺,
    )
end

abstract type GradNumericalFlux end
struct CentralNumericalFluxHigherOrder <: GradNumericalFlux end

function numerical_flux_higher_order!(
    ::CentralNumericalFluxHigherOrder,
    balance_law::BalanceLaw,
    hyperdiff::Vars{HD},
    normal_vector::SVector,
    lap⁻::Vars{GL},
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {HD, GL, S, A}
    G = normal_vector .* (parent(lap⁻) .+ parent(lap⁺))' ./ 2
    transform_post_gradient_laplacian!(
        balance_law,
        hyperdiff,
        Grad{GL}(G),
        state_conservative⁻,
        state_auxiliary⁻,
        t,
    )
end

function numerical_boundary_flux_higher_order!(
    numerical_flux::CentralNumericalFluxHigherOrder,
    balance_law::BalanceLaw,
    hyperdiff::Vars{HD},
    normal_vector::SVector,
    lap⁻::Vars{GL},
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    bctype,
    t,
) where {HD, GL, S, A}
    boundary_state!(
        numerical_flux,
        balance_law,
        state_conservative⁺,
        state_auxiliary⁺,
        lap⁺,
        normal_vector,
        state_conservative⁻,
        state_auxiliary⁻,
        lap⁻,
        bctype,
        t,
    )
    numerical_flux_higher_order!(
        numerical_flux,
        balance_law,
        hyperdiff,
        normal_vector,
        lap⁻,
        state_conservative⁻,
        state_auxiliary⁻,
        lap⁺,
        state_conservative⁺,
        state_auxiliary⁺,
        t,
    )
end

numerical_boundary_flux_second_order!(
    numerical_flux::CentralNumericalFluxSecondOrder,
    balance_law::BalanceLaw,
    states::NamedTuple,
    normal_vector::SVector,
    bctype,
    t,
    states1⁻::NamedTuple,
) = normal_boundary_flux_second_order!(
    numerical_flux,
    balance_law,
    states,
    normal_vector,
    bctype,
    t,
    states1⁻,
)

function normal_boundary_flux_second_order!(
    numerical_flux,
    balance_law::BalanceLaw,
    states::NamedTuple,
    normal_vector,
    bctype,
    t,
    states1⁻::NamedTuple,
)
    boundary_state!(
        numerical_flux,
        balance_law,
        states,
        normal_vector,
        bctype,
        t,
        states1⁻,
    )

    numerical_flux_second_order!(
        numerical_flux,
        balance_law,
        states,
        normal_vector,
        t,
    )
end

end
