"""
    module CplTestingBL
 Defines an equation set for testing coupling

 Defines kernels to evaluates RHS of

  d ϕ / dt = - div κ( ϕ_init ) grad ϕ

 subject to specified prescribed gradient or flux bc's

 - [`CplTestBL`](@ref)
     balance law struct created by this module

"""
module CplTestingBL

export CplTestBL
export PenaltyNumFluxDiffusive
export ExteriorBoundary
export CoupledPrimaryBoundary, CoupledSecondaryBoundary

const τ = 60 * 86400
using ClimateMachine.BalanceLaws:
    Auxiliary, BalanceLaw, Gradient, GradientFlux, Prognostic

import ClimateMachine.BalanceLaws:
    boundary_conditions,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    init_state_prognostic!,
    nodal_init_state_auxiliary!,
    source!,
    vars_state,
    wavespeed

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays

using ClimateMachine.DGMethods.NumericalFluxes:
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder,
    NumericalFluxFirstOrder,
    NumericalFluxSecondOrder,
    RusanovNumericalFlux

import ClimateMachine.DGMethods.NumericalFluxes:
    numerical_boundary_flux_second_order!, numerical_flux_second_order!


using ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays

"""
    CplTestBL
Type that holds specification for a diffusion equation balance law instance.
"""
struct CplTestBL{BLP, BCS} <: BalanceLaw
    bl_prop::BLP
    boundaryconditions::BCS
end

l_type = CplTestBL

"""
  Extend the NumericalFluxSecondOrder to include a penalty term numerical
  flux formulation.
"""
struct PenaltyNumFluxDiffusive <: NumericalFluxSecondOrder end

"""
  Set a default set of properties and their default values
  - init_aux_geom   :: function to initialize geometric terms stored in aux.
  - init_theta      :: function to set initial θ values.
  - source_theta    :: function to add a source term to θ.
  - calc_kappa_diff :: function to set diffusion coeffiecient(s).
  - get_wavespeed   :: function to return a wavespeed for Rusanov computations (there aren't any in this model)
  - get_penalty_tau :: function to set timescale on which to bring state+ and state- together
  - theta_shadow_boundary_flux :: function to set boundary flux into shadow variable for passing to coupler
"""
function prop_defaults()
    bl_prop = NamedTuple()

    function init_aux_geom(npt, elnum, x, y, z)
        return npt, elnum, x, y, z
    end
    # init_aux_geom(_...)=(return 0., 0., 0., 0., 0.)
    bl_prop = (bl_prop..., init_aux_geom = init_aux_geom)

    init_theta(_...) = (return 0.0)
    bl_prop = (bl_prop..., init_theta = init_theta)

    source_theta(_...) = (return 0.0)
    bl_prop = (bl_prop..., source_theta = source_theta)

    calc_kappa_diff(_...) = (return 0.0, 0.0, 0.0)
    bl_prop = (bl_prop..., calc_kappa_diff = calc_kappa_diff)

    get_wavespeed(_...) = (return 0.0)
    bl_prop = (bl_prop..., get_wavespeed = get_wavespeed)

    get_penalty_tau(_...) = (return 1.0)
    bl_prop = (bl_prop..., get_penalty_tau = get_penalty_tau)

    theta_shadow_boundary_flux(_...) = (return 0.0)
    bl_prop =
        (bl_prop..., theta_shadow_boundary_flux = theta_shadow_boundary_flux)

    coupling_lambda(_...) = (return 0.0)
    bl_prop = (bl_prop..., coupling_lambda = coupling_lambda)

    bl_prop = (bl_prop..., LAW = CplTestBL)
end

"""
  Declare single prognostic state variable, θ
  and a shadow variable, θ_boundary_export, for accumulating boundary flux over a single
  timestep.

  "Shadow" variables XX_boundary_export will be used to capture boundary fluxes that we
  want to export to coupling as time integrals. We use a shadow variable
  because we want to integrate over whatever timestepper is being used.
  Eventually we should have the ability to potentially use a 2d field here.
  The shadow variable needs to be zeroed at the start of each
  coupling cycle for a component.
"""
function vars_state(bl::l_type, ::Prognostic, FT)
    @vars begin
        θ::FT
        F_accum::FT # accumulated flux across boundary (atmosphere export)
    end
end

"""
  Auxiliary state variable symbols for array index and real world coordinates and for
  θ value at reference time used to compute κ.
"""
function vars_state(bl::l_type, ::Auxiliary, FT)
    @vars begin
        npt::Int
        elnum::Int
        xc::FT
        yc::FT
        zc::FT
        θⁱⁿⁱᵗ::FT
        θ_secondary::FT # stores opposite face for primary (atmospheric import)
        F_prescribed::FT # stores prescribed flux for secondary (ocean import)
    end
end

"""
  Gradient computation stage input (and output) variable symbols
"""
function vars_state(bl::l_type, ::Gradient, FT)
    @vars begin
        ∇θ::FT
        ∇θⁱⁿⁱᵗ::FT
    end
end

"""
  Flux due to gradient accumulation stage variable symbols
"""
function vars_state(bl::l_type, ::GradientFlux, FT)
    @vars begin
        κ∇θ::SVector{3, FT}
    end
end

"""
  Initialize prognostic state variables
"""
function init_state_prognostic!(
    bl::l_type,
    Q::Vars,
    A::Vars,
    geom::LocalGeometry,
    FT,
)
    npt = getproperty(geom, :n)
    elnum = getproperty(geom, :e)
    x = geom.coord[1]
    y = geom.coord[2]
    z = geom.coord[3]
    Q.θ = bl.bl_prop.init_theta(x, y, z, npt, elnum)
    Q.F_accum = 0
    nothing
end

"""
  Initialize auxiliary state variables
"""
function nodal_init_state_auxiliary!(
    bl::l_type,
    A::Vars,
    tmp::Vars,
    geom::LocalGeometry,
    _...,
)
    npt = getproperty(geom, :n)
    elnum = getproperty(geom, :e)
    x = geom.coord[1]
    y = geom.coord[2]
    z = geom.coord[3]
    A.npt, A.elnum, A.xc, A.yc, A.zc =
        bl.bl_prop.init_aux_geom(npt, elnum, x, y, z)
    A.θⁱⁿⁱᵗ = 0
    A.θ_secondary = 0
    A.F_prescribed = 0
    nothing
end

#====

Atmos

----

Land


====#




"""
  Set any source terms for prognostic state external sources.
  Also use to record boundary flux terms into shadow variables
  for export to coupler.
"""
function source!(bl::l_type, S::Vars, Q::Vars, G::Vars, A::Vars, _...)
    #S.θ=bl.bl_prop.source_theta(Q.θ,A.npt,A.elnum,A.xc,A.yc,A.zc,A.θ_secondary)
    # Record boundary condition fluxes as needed by adding to shadow
    # prognostic variable
    S.F_accum = (Q.θ - A.θ_secondary) * bl.bl_prop.coupling_lambda()
    nothing
end

"""
  No flux first order for diffusion equation, but we must define a stub.
"""
function flux_first_order!(bl::l_type, _...)
    nothing
end

"""
  Set values to have gradients computed.
"""
function compute_gradient_argument!(bl::l_type, G::Vars, Q::Vars, A::Vars, t)
    G.∇θ = Q.θ
    G.∇θⁱⁿⁱᵗ = A.θⁱⁿⁱᵗ
    nothing
end

"""
  Compute diffusivity tensor times computed gradient to give net gradient flux.
"""
function compute_gradient_flux!(
    bl::l_type,
    GF::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)
    # "Non-linear" form (for time stepped)
    ### κ¹,κ²,κ³=bl.bl_prop.calc_kappa_diff(G.∇θ,A.npt,A.elnum,A.xc,A.yc,A.zc)
    # "Linear" form (for implicit)
    κ¹, κ², κ³ =
        bl.bl_prop.calc_kappa_diff(G.∇θⁱⁿⁱᵗ, A.npt, A.elnum, A.xc, A.yc, A.zc)
    # Maybe I should pass both G.∇θ and G.∇θⁱⁿⁱᵗ?
    GF.κ∇θ = Diagonal(@SVector([κ¹, κ², κ³])) * G.∇θ
    nothing
end

"""
  Pass flux components for second order term into update kernel.
"""
function flux_second_order!(
    bl::l_type,
    F::Grad,
    Q::Vars,
    GF::Vars,
    H::Vars,
    A::Vars,
    t,
)
    F.θ += GF.κ∇θ
    nothing
end

# Boundary conditions

"""
  Define boundary condition flags/types to iterate over, for now keep it simple.
"""
function boundary_conditions(bl::l_type, _...)
    bl.boundaryconditions
end

"""
  Set any first order numerical flux to null
  NumericalFluxFirstOrder is an abstract type that currently generalizes
  RusanovNumericalFlux, CentralNumericalFluxFirstOrder, RoeNumericalFlux, HLLCNumericalFlux.
"""
# No first order fluxes so numerical flux needed. NumericalFluxFirstOrder
function boundary_state!(
    nF::NumericalFluxFirstOrder,
    bc,
    bl::l_type,
    Q⁺::Vars,
    A⁺::Vars,
    n,
    Q⁻::Vars,
    A⁻::Vars,
    t,
    _...,
)
    nothing
end


## ExteriorBoundary
# flux is 0 across the boundary
struct ExteriorBoundary end

"""
  Zero normal gradient boundary condition.
"""
function boundary_state!(
    nF::Union{CentralNumericalFluxGradient},
    bc::ExteriorBoundary,
    bl::l_type,
    Q⁺::Vars,
    A⁺::Vars,
    n,
    Q⁻::Vars,
    A⁻::Vars,
    t,
    _...,
)
    Q⁺.θ = Q⁻.θ
    nothing
end
function numerical_boundary_flux_second_order!(
    numerical_flux::NumericalFluxSecondOrder,
    bctype::ExteriorBoundary,
    balance_law::l_type,
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
) where {S, D, A, HD}
    fluxᵀn.θ = 0
end


## CoupledPrimaryBoundary
# compute flux based on opposite face
# also need to accumulate net flux across boundary
struct CoupledPrimaryBoundary end


function boundary_state!(
    nF::Union{CentralNumericalFluxGradient},
    bc::CoupledPrimaryBoundary,
    bl::l_type,
    Q⁺::Vars,
    A⁺::Vars,
    n,
    Q⁻::Vars,
    A⁻::Vars,
    t,
    _...,
)
    Q⁺.θ = Q⁻.θ # Q⁺.θ=A⁺.θ_secondary
    nothing
end
function numerical_boundary_flux_second_order!(
    numerical_flux::NumericalFluxSecondOrder,
    bctype::CoupledPrimaryBoundary,
    balance_law::l_type,
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
) where {S, D, A, HD}

    fluxᵀn.θ =
        (state_prognostic⁻.θ - state_auxiliary⁺.θ_secondary) *
        balance_law.bl_prop.coupling_lambda() # W/m^2
end


# θ: J / m^3
# ∇θ: J / m^4
# κ: (m^2/s)
# κ∇θ: J/(m^2 s) = W/m^2 = J/m^3 * m/s
#

#  - clean up and write primer
#  - imports and exports
#  - add boundary auxiliary interface
#  - add boundary flux accumulation interface
#  - vector quantity
#  - run on sphere to check normal terms are all correct
#  - single stack integration
#  - Held-Suarez with wind stress and heat

## CoupledSecondaryBoundary
# use prescribed flux computed in primary
struct CoupledSecondaryBoundary end
function boundary_state!(
    nF::Union{CentralNumericalFluxGradient},
    bc::CoupledSecondaryBoundary,
    bl::l_type,
    Q⁺::Vars,
    A⁺::Vars,
    n,
    Q⁻::Vars,
    A⁻::Vars,
    t,
    _...,
)
    Q⁺.θ = Q⁻.θ
    nothing
end
function numerical_boundary_flux_second_order!(
    numerical_flux::NumericalFluxSecondOrder,
    bctype::CoupledSecondaryBoundary,
    balance_law::l_type,
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
) where {S, D, A, HD}
    fluxᵀn.θ = -state_auxiliary⁺.F_prescribed
end

function wavespeed(bl::l_type, _...)
    # Used in Rusanov term.
    # Only active if there is a flux first order term?
    bl.bl_prop.get_wavespeed()
end

"""
  Penalty flux formulation of second order numerical flux. This formulation
  computes the CentralNumericalFluxSecondOrder term first (which is just the average
  of the + and - fluxes and an edge), and then adds a "penalty" flux that relaxes
  the edge state + and - toward each other.
"""
function numerical_flux_second_order!(
    ::PenaltyNumFluxDiffusive,
    bl::l_type,
    fluxᵀn::Vars{S},
    n::SVector,
    state⁻::Vars{S},
    diff⁻::Vars{D},
    hyperdiff⁻::Vars{HD},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    diff⁺::Vars{D},
    hyperdiff⁺::Vars{HD},
    aux⁺::Vars{A},
    t,
) where {S, HD, D, A}

    numerical_flux_second_order!(
        CentralNumericalFluxSecondOrder(),
        bl,
        fluxᵀn,
        n,
        state⁻,
        diff⁻,
        hyperdiff⁻,
        aux⁻,
        state⁺,
        diff⁺,
        hyperdiff⁺,
        aux⁺,
        t,
    )

    Fᵀn = parent(fluxᵀn)
    FT = eltype(Fᵀn)
    tau = bl.bl_prop.get_penalty_tau()
    Fᵀn .+= tau * (parent(state⁻) - parent(state⁺))
end


end
