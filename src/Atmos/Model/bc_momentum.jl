abstract type MomentumBC end
abstract type MomentumDragBC end
import ..BalanceLaws: bc_string

"""
    Impenetrable(drag::MomentumDragBC) :: MomentumBC

Defines an impenetrable wall model for momentum. This implies:
  - no flow in the direction normal to the boundary, and
  - flow parallel to the boundary is subject to the `drag` condition.
"""
struct Impenetrable{D <: MomentumDragBC} <: MomentumBC
    drag::D
end
prognostic_vars(::Impenetrable) = (Momentum(),)

"""
    FreeSlip() :: MomentumDragBC

No surface drag on momentum parallel to the boundary.
"""
struct FreeSlip <: MomentumDragBC end

function boundary_value(
    ::Momentum,
    ::Impenetrable{FreeSlip},
    atmos,
    args,
    ::NF1,
)
    @unpack state⁻, n = args
    return state⁻.ρu - 2 * dot(state⁻.ρu, n) .* SVector(n)
end
function boundary_value(
    ::Momentum,
    ::Impenetrable{FreeSlip},
    atmos,
    args,
    ::NF∇,
)
    @unpack state⁻, n = args
    return state⁻.ρu - dot(state⁻.ρu, n) .* SVector(n)
end
function boundary_flux(::Momentum, ::Impenetrable{FreeSlip}, atmos, args, ::NF2)
    return DefaultBCFlux()
end

function atmos_momentum_boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::Impenetrable{FreeSlip},
    atmos,
    state⁺,
    args,
)
    @unpack state⁻, n = args
    state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
end
function atmos_momentum_boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::Impenetrable{FreeSlip},
    atmos,
    state⁺,
    args,
)
    @unpack state⁻, n = args
    state⁺.ρu -= dot(state⁻.ρu, n) .* SVector(n)
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{FreeSlip},
    atmos,
    _...,
) end



"""
    NoSlip() :: MomentumDragBC

Zero momentum at the boundary.
"""
struct NoSlip <: MomentumDragBC end

function boundary_value(::Momentum, ::Impenetrable{NoSlip}, atmos, args, ::NF1)
    @unpack state⁻ = args
    return -state⁻.ρu
end
function boundary_value(::Momentum, ::Impenetrable{NoSlip}, atmos, args, ::NF∇)
    @unpack state⁻ = args
    return zero(state⁻.ρu)
end

function atmos_momentum_boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc_momentum::Impenetrable{NoSlip},
    atmos,
    state⁺,
    args,
)
    @unpack state⁻ = args
    state⁺.ρu = -state⁻.ρu
end
function atmos_momentum_boundary_state!(
    nf::NumericalFluxGradient,
    bc_momentum::Impenetrable{NoSlip},
    atmos,
    state⁺,
    args,
)
    state⁺.ρu = zero(state⁺.ρu)
end
function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{NoSlip},
    atmos,
    _...,
) end


"""
    DragLaw(fn) :: MomentumDragBC

Drag law for momentum parallel to the boundary. The drag coefficient is
`C = fn(state, aux, t, normu_int_tan)`, where `normu_int_tan` is the internal speed
parallel to the boundary.
`_int` refers to the first interior node.
"""
struct DragLaw{FN} <: MomentumDragBC
    fn::FN
end

function boundary_value(
    pv::Momentum,
    bc::Impenetrable{DL},
    atmos,
    args,
    nf::Union{NF1, NF∇},
) where {DL <: DragLaw}
    return boundary_value(pv, Impenetrable(FreeSlip()), atmos, args, nf)
end

function atmos_momentum_boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc_momentum::Impenetrable{DL},
    atmos,
    state⁺,
    args,
) where {DL <: DragLaw}
    atmos_momentum_boundary_state!(
        nf,
        Impenetrable(FreeSlip()),
        atmos,
        state⁺,
        args,
    )
end


function bc_precompute(
    bc::Impenetrable{DL},
    atmos,
    args,
    nf::NF2,
) where {DL <: DragLaw}
    @unpack state⁻, state_int⁻, aux⁻, t, n⁻ = args
    u1⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = u1⁻ - dot(u1⁻, n⁻) .* SVector(n⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    # NOTE: difference from design docs since normal points outwards
    C = bc.drag.fn(state⁻, aux⁻, t, normu_int⁻_tan)
    τn = C * normu_int⁻_tan * u_int⁻_tan
    return (; τn)
end

function boundary_flux(
    pv::Momentum,
    bc::Impenetrable{DL},
    atmos,
    args,
    nf::NF2,
) where {DL <: DragLaw}
    @unpack state⁻, precomputed = args
    @unpack τn = precomputed[bc]
    # both sides involve projections of normals, so signs are consistent
    return state⁻.ρ * τn
end

function atmos_momentum_normal_boundary_flux_second_order!(
    nf,
    bc_momentum::Impenetrable{DL},
    atmos,
    fluxᵀn,
    args,
) where {DL <: DragLaw}
    @unpack state⁻, aux⁻, n⁻, t, aux_int⁻, state_int⁻ = args

    u1⁻ = state_int⁻.ρu / state_int⁻.ρ
    u_int⁻_tan = u1⁻ - dot(u1⁻, n⁻) .* SVector(n⁻)
    normu_int⁻_tan = norm(u_int⁻_tan)
    # NOTE: difference from design docs since normal points outwards
    C = bc_momentum.drag.fn(state⁻, aux⁻, t, normu_int⁻_tan)
    τn = C * normu_int⁻_tan * u_int⁻_tan
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρu += state⁻.ρ * τn
end

bc_string(::Impenetrable{FreeSlip}) = "Impenetrable{FreeSlip}"
bc_string(::Impenetrable{NoSlip}) = "Impenetrable{NoSlip}"
bc_string(::Impenetrable{DL}) where {DL <: DragLaw} = "Impenetrable{DragLaw}"
