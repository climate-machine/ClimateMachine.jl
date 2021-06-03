module NumericalFluxes

export NumericalFluxGradient,
    NumericalFluxFirstOrder,
    NumericalFluxSecondOrder,
    RusanovNumericalFlux,
    RoeNumericalFlux,
    HLLCNumericalFlux,
    RoeNumericalFluxMoist,
    CentralNumericalFluxGradient,
    CentralNumericalFluxFirstOrder,
    CentralNumericalFluxSecondOrder,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxHigherOrder,
    LMARSNumericalFlux


using StaticArrays, LinearAlgebra
using ClimateMachine.VariableTemplates
using KernelAbstractions.Extras: @unroll
using ...BalanceLaws
import ...BalanceLaws:
    vars_state,
    boundary_state!,
    wavespeed,
    total_flux_first_order!,
    flux_second_order!,
    compute_gradient_flux!,
    compute_gradient_argument!,
    transform_post_gradient_laplacian!,
    boundary_conditions

"""
    NumericalFluxGradient

Any `P <: NumericalFluxGradient` should define methods for:

    numerical_flux_gradient!(
        gnf::P,
        balance_law::BalanceLaw,
        diffF, n⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        t
    )

    numerical_boundary_flux_gradient!(
        gnf::P,
        balance_law::BalanceLaw,
        local_state_gradient_flux,
        n⁻,
        local_transform⁻, local_state_prognostic⁻, local_state_auxiliary⁻,
        local_transform⁺, local_state_prognostic⁺, local_state_auxiliary⁺,
        bctype,
        t
    )

"""
abstract type NumericalFluxGradient end

"""
    CentralNumericalFluxGradient <: NumericalFluxGradient

"""
struct CentralNumericalFluxGradient <: NumericalFluxGradient end

function numerical_flux_gradient!(
    ::CentralNumericalFluxGradient,
    balance_law::BalanceLaw,
    transform_gradient::MMatrix,
    normal_vector::AbstractArray,
    state_gradient⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_gradient⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)

    transform_gradient .=
        SVector(normal_vector) .* (state_gradient⁺ .+ state_gradient⁻)' ./ 2
end

function numerical_boundary_flux_gradient!(
    numerical_flux::CentralNumericalFluxGradient,
    bctype,
    balance_law::BalanceLaw,
    transform_gradient::MMatrix,
    normal_vector::SVector,
    state_gradient⁻::Vars{T},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_gradient⁺::Vars{T},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {D, T, S, A}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        aux1⁻,
    )

    compute_gradient_argument!(
        balance_law,
        state_gradient⁺,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )
    transform_gradient .= normal_vector .* parent(state_gradient⁺)'
end

"""
    NumericalFluxFirstOrder

Any `N <: NumericalFluxFirstOrder` should define the a method for

    numerical_flux_first_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qaux⁻,
        Q⁺, Qaux⁺,
        t
    )

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_first_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qaux⁻,
        Q⁺, Qaux⁺,
        bctype, t
    )

"""
abstract type NumericalFluxFirstOrder end

function numerical_flux_first_order! end

function numerical_boundary_flux_first_order!(
    numerical_flux::NumericalFluxFirstOrder,
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}

    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        aux1⁻,
    )

    numerical_flux_first_order!(
        numerical_flux,
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
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
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    fluxᵀn = parent(fluxᵀn)
    wavespeed⁻ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    wavespeed⁺ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    max_wavespeed = max.(wavespeed⁻, wavespeed⁺)
    penalty =
        max_wavespeed .* (parent(state_prognostic⁻) - parent(state_prognostic⁺))

    # TODO: should this operate on ΔQ or penalty?
    update_penalty!(
        numerical_flux,
        balance_law,
        normal_vector,
        max_wavespeed,
        Vars{S}(penalty),
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )

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
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    total_flux_first_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )

    flux⁺ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁺, -zero(FT))
    total_flux_first_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector / 2)
end

"""
    RoeNumericalFlux() <: NumericalFluxFirstOrder

A numerical flux based on the approximate Riemann solver of Roe

# Usage

    RoeNumericalFlux()

Requires a custom implementation for the balance law.
"""
struct RoeNumericalFlux <: NumericalFluxFirstOrder end

"""
    HLLCNumericalFlux() <: NumericalFluxFirstOrder

A numerical flux based on the approximate Riemann solver of the
HLLC method. The HLLC flux is a modification of the Harten, Lax, van-Leer
(HLL) flux, where an additional contact property is introduced in order
to restore missing rarefraction waves. The HLLC flux requires
model-specific information, hence it requires a custom implementation
based on the underlying balance law.

# Usage

    HLLCNumericalFlux()

Requires a custom implementation for the balance law.

 - [Toro2013](@cite)
"""
struct HLLCNumericalFlux <: NumericalFluxFirstOrder end


"""
    LMARSNumericalFlux <: NumericalFluxFirstOrder
Low Mach Number Approximate Riemann Solver. Upwind biased
first order flux function. 

- [Chen2013](@cite)
"""
struct LMARSNumericalFlux <: NumericalFluxFirstOrder end

"""
    RoeNumericalFluxMoist <: NumericalFluxFirstOrder

A moist implementation of the numerical flux based on the approximate Riemann solver of Roe

Requires a custom implementation for the balance law.
"""
struct RoeNumericalFluxMoist <: NumericalFluxFirstOrder
    " set to true for low Mach number correction"
    LM::Bool
    " set to true for Hartman Hyman correction"
    HH::Bool
    " set to true for LeVeque correction"
    LV::Bool
    " set to true for Positivity preserving LeVeque Correction"
    LVPP::Bool
end

RoeNumericalFluxMoist(;
    LM::Bool = false,
    HH::Bool = false,
    LV::Bool = false,
    LVPP::Bool = false,
) = RoeNumericalFluxMoist(LM, HH, LV, LVPP)

"""
dispatch type for entropy conservative numerical fluxes (these are balance law
specific)
"""
struct EntropyConservative <: NumericalFluxFirstOrder end

"""
    numerical_volume_conservative_flux_first_order!(
        numflux::NumericalFluxFirstOrder,
        balancelaw::BalanceLaw,
        flux::Grad,
        state_1::Vars,
        aux_1::Vars,
        state_2::Vars,
        aux_2::Vars,
    )
Two point flux for use in the volume of entropy stable discretizations. Should
compute and store the conservative (symmetric) part of an entropy stable
flux-fluctuation and store the three vector components back to `flux`.
This should be symmetric in the sense that swapping the place of `state_1`,
`aux_1` with `state_2`, `aux_2` should result in the same flux.
This should be implemented with addition assignment.
Each balance law must implement a concrete implementation of this function.
"""
function numerical_volume_conservative_flux_first_order!(
    numflux::NumericalFluxFirstOrder,
    balancelaw::BalanceLaw,
    flux::AbstractArray{FT, 2},
    state_1::AbstractArray{FT, 1},
    aux_1::AbstractArray{FT, 1},
    state_2::AbstractArray{FT, 1},
    aux_2::AbstractArray{FT, 1},
) where {FT}
    numerical_volume_conservative_flux_first_order!(
        numflux,
        balancelaw,
        Grad{vars_state(balancelaw, Prognostic(), FT)}(flux),
        Vars{vars_state(balancelaw, Prognostic(), FT)}(state_1),
        Vars{vars_state(balancelaw, Auxiliary(), FT)}(aux_1),
        Vars{vars_state(balancelaw, Prognostic(), FT)}(state_2),
        Vars{vars_state(balancelaw, Auxiliary(), FT)}(aux_2),
    )
end
numerical_volume_conservative_flux_first_order!(::Nothing, _...) = nothing

"""
    numerical_volume_fluctuation_flux_first_order!(
        numflux::NumericalFluxFirstOrder,
        balancelaw::BalanceLaw,
        flux::Grad,
        state_1::Vars,
        aux_1::Vars,
        state_2::Vars,
        aux_2::Vars,
    )
Two point fluctuation flux for use in the volume of entropy stable
discretizations. Should compute and store the non-conservative (possibly
non-symmetric) part of an entropy stable flux-fluctuation and store the three
vector components back to `flux`.
This can be non-symmetric in the sense that swapping the place of `state_1`,
`aux_1` with `state_2`, `aux_2` can result in a different flux.
This should be implemented with addition assignment.
Each balance law must implement a concrete implementation of this function.
"""
function numerical_volume_fluctuation_flux_first_order!(
    numflux::NumericalFluxFirstOrder,
    balancelaw::BalanceLaw,
    fluctuation::AbstractArray{FT, 2},
    state_1::AbstractArray{FT, 1},
    aux_1::AbstractArray{FT, 1},
    state_2::AbstractArray{FT, 1},
    aux_2::AbstractArray{FT, 1},
) where {FT}
    numerical_volume_fluctuation_flux_first_order!(
        numflux,
        balancelaw,
        Grad{vars_state(balancelaw, Prognostic(), FT)}(fluctuation),
        Vars{vars_state(balancelaw, Prognostic(), FT)}(state_1),
        Vars{vars_state(balancelaw, Auxiliary(), FT)}(aux_1),
        Vars{vars_state(balancelaw, Prognostic(), FT)}(state_2),
        Vars{vars_state(balancelaw, Auxiliary(), FT)}(aux_2),
    )
end
numerical_volume_fluctuation_flux_first_order!(::Nothing, _...) = nothing

"""
    numerical_volume_flux_first_order!(
        numflux::NumericalFluxFirstOrder,
        balancelaw::BalanceLaw,
        flux::AbstractArray{FT, 2},
        state_1::AbstractArray{FT, 1},
        aux_1::AbstractArray{FT, 1},
        state_2::AbstractArray{FT, 1},
        aux_2::AbstractArray{FT, 1},
    )
Convenience function which calls
  `numerical_volume_conservative_flux_first_order!`
followed by
  `numerical_volume_fluctuation_flux_first_order!`
"""
function numerical_volume_flux_first_order!(
    numflux::NumericalFluxFirstOrder,
    balancelaw::BalanceLaw,
    flux::AbstractArray{FT, 2},
    state_1::AbstractArray{FT, 1},
    aux_1::AbstractArray{FT, 1},
    state_2::AbstractArray{FT, 1},
    aux_2::AbstractArray{FT, 1},
) where {FT}
    numerical_volume_conservative_flux_first_order!(
        numflux,
        balancelaw,
        flux,
        state_1,
        aux_1,
        state_2,
        aux_2,
    )
    numerical_volume_fluctuation_flux_first_order!(
        numflux,
        balancelaw,
        flux,
        state_1,
        aux_1,
        state_2,
        aux_2,
    )
end
numerical_volume_flux_first_order!(::Nothing, _...) = nothing

function numerical_flux_first_order!(
    numerical_flux::EntropyConservative,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    # Let's just work with straight arrays!
    fluxᵀn = parent(fluxᵀn)
    state_1 = parent(state_prognostic⁻)
    aux_1 = parent(state_auxiliary⁻)
    state_2 = parent(state_prognostic⁺)
    aux_2 = parent(state_auxiliary⁺)

    # create the storage for the volume flux
    num_state = length(fluxᵀn)
    FT = eltype(fluxᵀn)
    H = MArray{Tuple{3, num_state}, FT}(undef)
    fill!(H, -zero(FT))

    # Compute the volume flux
    numerical_volume_flux_first_order!(
        numerical_flux,
        balance_law,
        H,
        state_1,
        aux_1,
        state_2,
        aux_2,
    )

    # Multiply in the normal
    @inbounds fluxᵀn .+=
        normal_vector[1] * H[1, :] +
        normal_vector[2] * H[2, :] +
        normal_vector[3] * H[3, :]
end

# Convenience function for entropy stable discretizations
"""
    ave(a, b)
This computes the mean
    ave(a, b) = (a + b) / 2
"""
ave(a, b) = (a + b) / 2

# Convenience function for entropy stable discretizations
"""
    logave(a, b)
This computes the logarithmic mean
    logave(a, b) = (a - b) / (log(a) - log(b))
in a numerically stable way using the method in Appendix B. of Ishail and Roe
<doi:10.1016/j.jcp.2009.04.021>.
"""
function logave(a, b)
    ζ = a / b
    f = (ζ - 1) / (ζ + 1)
    u = f^2
    ϵ = eps(eltype(u))

    if u < ϵ
        F = @evalpoly(u, one(u), one(u) / 3, one(u) / 5, one(u) / 7, one(u) / 9)
    else
        F = log(ζ) / 2f
    end

    (a + b) / 2F
end

"""
    NumericalFluxSecondOrder

Any `N <: NumericalFluxSecondOrder` should define the a method for

    numerical_flux_second_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        t
    )

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive diffusive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux_second_order!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux,
        normal_vector⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        bctype,
        t
    )

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
    fluxᵀn::Vars{S},
    normal_vector⁻::SVector,
    state_prognostic⁻::Vars{S},
    state_gradient_flux⁻::Vars{D},
    state_hyperdiffusive⁻::Vars{HD},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_gradient_flux⁺::Vars{D},
    state_hyperdiffusive⁺::Vars{HD},
    state_auxiliary⁺::Vars{A},
    t,
) where {S, D, HD, A}

    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_second_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_hyperdiffusive⁻,
        state_auxiliary⁻,
        t,
    )

    flux⁺ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁺, -zero(FT))
    flux_second_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )

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
        (parent(grad⁺) .+ parent(grad⁻))' * (normal_vector / 2)
end

function numerical_boundary_flux_divergence!(
    numerical_flux::CentralNumericalFluxDivergence,
    bctype,
    balance_law::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    state_auxiliary⁻::Vars{A},
    grad⁺::Grad{GL},
    state_auxiliary⁺::Vars{A},
    t,
) where {GL, A}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        grad⁺,
        state_auxiliary⁺,
        normal_vector,
        grad⁻,
        state_auxiliary⁻,
        t,
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
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {HD, GL, S, A}
    G = normal_vector .* (parent(lap⁺) .- parent(lap⁻))' ./ 2
    transform_post_gradient_laplacian!(
        balance_law,
        hyperdiff,
        Grad{GL}(G),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
    )
end

function numerical_boundary_flux_higher_order!(
    numerical_flux::CentralNumericalFluxHigherOrder,
    bctype,
    balance_law::BalanceLaw,
    hyperdiff::Vars{HD},
    normal_vector::SVector,
    lap⁻::Vars{GL},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {HD, GL, S, A}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        lap⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        lap⁻,
        t,
    )
    numerical_flux_higher_order!(
        numerical_flux,
        balance_law,
        hyperdiff,
        normal_vector,
        lap⁻,
        state_prognostic⁻,
        state_auxiliary⁻,
        lap⁺,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )
end

numerical_boundary_flux_second_order!(
    numerical_flux::CentralNumericalFluxSecondOrder,
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_gradient_flux⁻::Vars{D},
    state_hyperdiffusive⁻::Vars{HD},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_gradient_flux⁺::Vars{D},
    state_hyperdiffusive⁺::Vars{HD},
    state_auxiliary⁺::Vars{A},
    t,
    state1⁻::Vars{S},
    diff1⁻::Vars{D},
    aux1⁻::Vars{A},
) where {S, D, HD, A} = normal_boundary_flux_second_order!(
    numerical_flux,
    bctype,
    balance_law,
    fluxᵀn,
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
)

function normal_boundary_flux_second_order!(
    numerical_flux,
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
) where {S}
    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux, -zero(FT))
    boundary_flux_second_order!(
        numerical_flux,
        bctype,
        balance_law,
        Grad{S}(flux),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_hyperdiffusive⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        diff1⁻,
        aux1⁻,
    )

    fluxᵀn .+= flux' * normal_vector
end

# This is the function that my be overloaded for flux-based BCs
function boundary_flux_second_order!(
    numerical_flux::NumericalFluxSecondOrder,
    bctype,
    balance_law,
    flux,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
)
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_hyperdiffusive⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        diff1⁻,
        aux1⁻,
    )
    flux_second_order!(
        balance_law,
        flux,
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )
end

# Array wrappers for numerical Fluxes
function numerical_flux_first_order!(
    numerical_flux_first_order,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    face_direction,
)
    FT = eltype(flux)
    numerical_flux_first_order!(
        numerical_flux_first_order,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
        face_direction,
    )
end

function numerical_boundary_flux_first_order!(
    numerical_flux_first_order,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    face_direction,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    FT = eltype(flux)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        numerical_boundary_flux_first_order!(
            numerical_flux_first_order,
            bc,
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
            SVector(normal_vector),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            face_direction,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
        )
    end d -> throw(BoundsError(bcs, bctag))
end

function numerical_flux_second_order!(
    numerical_flux_second_order,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
)
    FT = eltype(flux)
    numerical_flux_second_order!(
        numerical_flux_second_order,
        balance_law,
        Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
        SVector(normal_vector),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁻),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁻,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
        Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
        Vars{vars_state(balance_law, GradientFlux(), FT)}(state_gradient_flux⁺),
        Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
            state_hyperdiffusive⁺,
        ),
        Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
        t,
    )
end

function numerical_boundary_flux_second_order!(
    numerical_flux_second_order,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
    state_gradient_flux_bottom1::AbstractArray,
)
    FT = eltype(flux)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        numerical_boundary_flux_second_order!(
            numerical_flux_second_order,
            bc,
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
            SVector(normal_vector),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux⁻,
            ),
            Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                state_hyperdiffusive⁻,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux⁺,
            ),
            Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                state_hyperdiffusive⁺,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
        )
    end d -> throw(BoundsError(bcs, bctag))
end

function numerical_boundary_flux_gradient!(
    numerical_flux_gradient,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_gradient⁻::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_gradient⁺::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    FT = eltype(flux)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        # Computes G* incorporating boundary conditions
        numerical_boundary_flux_gradient!(
            numerical_flux_gradient,
            bc,
            balance_law,
            flux,
            SVector(normal_vector),
            Vars{vars_state(balance_law, Gradient(), FT)}(state_gradient⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Gradient(), FT)}(state_gradient⁺),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
        )
    end d -> throw(BoundsError(bcs, bctag))
end

end # end of module
