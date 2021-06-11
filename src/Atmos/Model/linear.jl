using CLIMAParameters.Planet: R_d, cv_d, T_0, e_int_v0, e_int_i0

"""
    linearized_air_pressure(ρ, ρe_tot, ρe_pot, ρq_tot=0, ρq_liq=0, ρq_ice=0)

The air pressure, linearized around a dry rest state, from the equation of state
(ideal gas law) where:

 - `ρ` (moist-)air density
 - `ρe_tot` total energy density
 - `ρe_pot` potential energy density
and, optionally,
 - `ρq_tot` total water density
 - `ρq_liq` liquid water density
 - `ρq_ice` ice density
"""
function linearized_air_pressure(
    param_set::AbstractParameterSet,
    ρ::FT,
    ρe_tot::FT,
    ρe_pot::FT,
    ρq_tot::FT = FT(0),
    ρq_liq::FT = FT(0),
    ρq_ice::FT = FT(0),
) where {FT <: Real, PS}
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    return ρ * _R_d * _T_0 +
           _R_d / _cv_d * (
        ρe_tot - ρe_pot - (ρq_tot - ρq_liq) * _e_int_v0 +
        ρq_ice * (_e_int_i0 + _e_int_v0)
    )
end

@inline linearized_pressure(atmos, state::Vars, aux::Vars) =
    linearized_pressure(
        moisture_model(atmos),
        parameter_set(atmos),
        atmos.orientation,
        state,
        aux,
    )

@inline function linearized_pressure(
    ::DryModel,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    ρe_pot = state.ρ * gravitational_potential(orientation, aux)
    return linearized_air_pressure(param_set, state.ρ, state.energy.ρe, ρe_pot)
end
@inline function linearized_pressure(
    ::EquilMoist,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    ρe_pot = state.ρ * gravitational_potential(orientation, aux)
    linearized_air_pressure(
        param_set,
        state.ρ,
        state.energy.ρe,
        ρe_pot,
        state.moisture.ρq_tot,
    )
end
@inline function linearized_pressure(
    ::NonEquilMoist,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    ρe_pot = state.ρ * gravitational_potential(orientation, aux)
    linearized_air_pressure(
        param_set,
        state.ρ,
        state.energy.ρe,
        ρe_pot,
        state.moisture.ρq_tot,
        state.moisture.ρq_liq,
        state.moisture.ρq_ice,
    )
end

abstract type AtmosLinearModel <: BalanceLaw end

"""
    vars_state(m::AtmosLinearModel, ::Prognostic, FT)

Conserved state variables (prognostic variables).

!!! warning

    `AtmosLinearModel` state ordering must be a contiguous subset of the initial
    state of `AtmosModel` since a shared state is used.
"""
function vars_state(lm::AtmosLinearModel, st::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        energy::vars_state(energy_model(lm.atmos), st, FT)
        turbulence::vars_state(turbulence_model(lm.atmos), st, FT)
        hyperdiffusion::vars_state(hyperdiffusion_model(lm.atmos), st, FT)
        moisture::vars_state(moisture_model(lm.atmos), st, FT)
    end
end
vars_state(lm::AtmosLinearModel, st::Auxiliary, FT) =
    vars_state(lm.atmos, st, FT)
vars_state(lm::AtmosLinearModel, ::Primitive, FT) =
    vars_state(lm, Prognostic(), FT)

function update_auxiliary_state!(
    dg::DGModel,
    lm::AtmosLinearModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return false
end
function flux_second_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
integral_set_auxiliary_state!(lm::AtmosLinearModel, aux::Vars, integ::Vars) =
    nothing
reverse_integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
reverse_integral_set_auxiliary_state!(
    lm::AtmosLinearModel,
    aux::Vars,
    integ::Vars,
) = nothing
function wavespeed(
    lm::AtmosLinearModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ref = aux.ref_state
    return soundspeed_air(parameter_set(lm.atmos), ref.T)
end

boundary_conditions(atmoslm::AtmosLinearModel) =
    (AtmosBC(atmoslm.atmos.physics), AtmosBC(atmoslm.atmos.physics))

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc,
    atmoslm::AtmosLinearModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    state_int⁻,
    aux_int⁻,
)

    args = (; aux⁺, state⁻, aux⁻, t, n, state_int⁻, aux_int⁻)

    set_boundary_values!(
        state⁺,
        atmoslm.atmos,
        nf,
        bc,
        args,
        prognostic_vars(atmoslm),
    )

    atmos_boundary_state!(nf, bc, atmoslm, state⁺, args)
end
function boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc,
    atmoslm::AtmosLinearModel,
    _...,
)
    nothing
end
init_state_auxiliary!(
    lm::AtmosLinearModel,
    aux::MPIStateArray,
    grid,
    direction,
) = nothing
init_state_prognostic!(
    lm::AtmosLinearModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
) = nothing


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticLinearModel(atmos::M) where {M}
        if reference_state(atmos) === NoReferenceState()
            error("AtmosAcousticLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end

function flux_first_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    tend = Flux{FirstOrder}()
    _args = (; state, aux, t, direction)

    # For some reason, we cannot call precompute, because
    # sometimes `state.ρ` is somehow `0`, which results in
    # `e_int = Inf` -> failed saturation adjustment.
    # TODO: look into this
    args = _args
    # args = merge(_args, (precomputed = precompute(lm.atmos, _args, tend),))
    flux.ρ = Σfluxes(Mass(), eq_tends(Mass(), lm, tend), lm, args)
    flux.ρu = Σfluxes(Momentum(), eq_tends(Momentum(), lm, tend), lm, args)
    flux.energy.ρe = Σfluxes(Energy(), eq_tends(Energy(), lm, tend), lm, args)
    nothing
end

struct AtmosAcousticGravityLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticGravityLinearModel(atmos::M) where {M}
        if reference_state(atmos) === NoReferenceState()
            error("AtmosAcousticGravityLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end

function source!(
    lm::AtmosLinearModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    ::NTuple{1, Dir},
) where {Dir <: Direction}

    tend = Source()
    _args = (; state, aux, t, direction = Dir, diffusive)

    # For some reason, we cannot call precompute, because
    # sometimes `state.ρ` is somehow `0`, which results in
    # `e_int = Inf` -> failed saturation adjustment.
    # TODO: look into this
    args = _args
    # args = merge(_args, (precomputed = precompute(lm.atmos, _args, tend),))

    # Sources for the linear atmos model only appear in the momentum equation
    source.ρu = Σsources(Momentum(), eq_tends(Momentum(), lm, tend), lm, args)
    nothing
end

function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFlux,
    balance_law::AtmosLinearModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    @assert moisture_model(balance_law.atmos) isa DryModel

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

    atmos = balance_law.atmos
    param_set = parameter_set(atmos)

    ρu⁻ = state_prognostic⁻.ρu

    ref_ρ⁻ = state_auxiliary⁻.ref_state.ρ
    ref_ρe⁻ = state_auxiliary⁻.ref_state.ρe
    ref_T⁻ = state_auxiliary⁻.ref_state.T
    ref_p⁻ = state_auxiliary⁻.ref_state.p
    ref_h⁻ = (ref_ρe⁻ + ref_p⁻) / ref_ρ⁻
    ref_c⁻ = soundspeed_air(param_set, ref_T⁻)

    pL⁻ = linearized_pressure(atmos, state_prognostic⁻, state_auxiliary⁻)

    ρu⁺ = state_prognostic⁺.ρu

    ref_ρ⁺ = state_auxiliary⁺.ref_state.ρ
    ref_ρe⁺ = state_auxiliary⁺.ref_state.ρe
    ref_T⁺ = state_auxiliary⁺.ref_state.T
    ref_p⁺ = state_auxiliary⁺.ref_state.p
    ref_h⁺ = (ref_ρe⁺ + ref_p⁺) / ref_ρ⁺
    ref_c⁺ = soundspeed_air(param_set, ref_T⁺)

    pL⁺ = linearized_pressure(atmos, state_prognostic⁺, state_auxiliary⁺)

    # not sure if arithmetic averages are a good idea here
    h̃ = (ref_h⁻ + ref_h⁺) / 2
    c̃ = (ref_c⁻ + ref_c⁺) / 2

    ΔpL = pL⁺ - pL⁻
    Δρuᵀn = (ρu⁺ - ρu⁻)' * normal_vector

    fluxᵀn.ρ -= ΔpL / 2c̃
    fluxᵀn.ρu -= c̃ * Δρuᵀn * normal_vector / 2
    fluxᵀn.energy.ρe -= h̃ * ΔpL / 2c̃
end

function numerical_flux_first_order!(
    ::HLLCNumericalFlux,
    balance_law::AtmosLinearModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    # There is no intermediate speed for the AtmosLinearModel.
    # As a result, HLLC simplifies to Rusanov.
    numerical_flux_first_order!(
        RusanovNumericalFlux(),
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
function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFluxMoist,
    balance_law::AtmosLinearModel,
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

    atmos = balance_law.atmos
    param_set = parameter_set(atmos)
    FT = eltype(aux)
    ρu⁻ = state_prognostic⁻.ρu

    ref_ρ⁻ = state_auxiliary⁻.ref_state.ρ
    ref_ρe⁻ = state_auxiliary⁻.ref_state.ρe
    ref_T⁻ = state_auxiliary⁻.ref_state.T
    ref_q⁻ = state_auxiliary⁻.ref_state.ρq_tot / ref_ρ⁻
    ref_qliq⁻ = state_auxiliary⁻.ref_state.ρq_liq / ref_ρ⁻
    ref_qice⁻ = state_auxiliary⁻.ref_state.ρq_ice / ref_ρ⁻
    ref_p⁻ = state_auxiliary⁻.ref_state.p
    ref_q_pt⁻ = PhasePartition(ref_q⁻, ref_qliq⁻, ref_qice⁻)
    _R_m⁻ = gas_constant_air(param_set, ref_q_pt⁻)
    ref_h⁻ = total_specific_enthalpy(ref_ρe⁻, _R_m⁻, ref_T⁻)

    ref_c⁻ = soundspeed_air(param_set, ref_T⁻, ref_q_pt⁻)

    pL⁻ = linearized_pressure(atmos, state_prognostic⁻, state_auxiliary⁻)

    ρu⁺ = state_prognostic⁺.ρu

    ref_ρ⁺ = state_auxiliary⁺.ref_state.ρ
    ref_ρe⁺ = state_auxiliary⁺.ref_state.ρe
    ref_T⁺ = state_auxiliary⁺.ref_state.T
    ref_q⁺ = state_auxiliary⁺.ref_state.ρq_tot / ref_ρ⁺
    ref_qliq⁺ = state_auxiliary⁺.ref_state.ρq_liq / ref_ρ⁺
    ref_qice⁺ = state_auxiliary⁺.ref_state.ρq_ice / ref_ρ⁺
    ref_p⁺ = state_auxiliary⁺.ref_state.p
    ref_q_pt⁺ = PhasePartition(ref_q⁺, ref_qliq⁺, ref_qice⁺)
    _R_m⁺ = gas_constant_air(param_set, ref_q_pt⁺)
    ref_h⁺ = total_specific_enthalpy(ref_ρe⁺, _R_m⁺, ref_T⁺)
    ref_c⁺ = soundspeed_air(param_set, ref_T⁺, ref_q_pt⁺)
    pL⁺ = linearized_pressure(atmos, state_prognostic⁺, state_auxiliary⁺)

    # not sure if arithmetic averages are a good idea here
    ref_h̃ = (ref_h⁻ + ref_h⁺) / 2
    ref_c̃ = (ref_c⁻ + ref_c⁺) / 2
    ref_qt = (ref_q⁺ + ref_q⁻) / 2

    ΔpL = pL⁺ - pL⁻
    Δρuᵀn = (ρu⁺ - ρu⁻)' * normal_vector

    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _cv_d::FT = cv_d(param_set)
    Φ = gravitational_potential(balance_law.atmos, state_auxiliary⁻)
    # guaranteed to be random
    ω = FT(π) / 3
    δ = FT(π) / 5
    random_unit_vector = SVector(sin(ω) * cos(δ), cos(ω) * cos(δ), sin(δ))
    # tangent space basis
    τ1 = random_unit_vector × normal_vector
    τ2 = τ1 × normal_vector


    ũc̃⁻ = ref_c̃ * normal_vector
    ũc̃⁺ = -ref_c̃ * normal_vector
    Λ = SDiagonal(
        abs(0 - ref_c̃),
        abs(0),
        abs(0),
        abs(0),
        abs(0 + ref_c̃),
        abs(0),
    )
    M = hcat(
        SVector(1, ũc̃⁺[1], ũc̃⁺[2], ũc̃⁺[3], ref_h̃, ref_qt),
        SVector(0, τ1[1], τ1[2], τ1[3], 0, 0),
        SVector(0, τ2[1], τ2[2], τ2[3], 0, 0),
        SVector(1, 0, 0, 0, Φ - _T_0 * _cv_d, 0),
        SVector(1, ũc̃⁻[1], ũc̃⁻[2], ũc̃⁻[3], ref_h̃, ref_qt),
        SVector(0, 0, 0, 0, _e_int_v0, 1),
    )
    Δρ = state_prognostic⁺.ρ - state_prognostic⁻.ρ
    Δρu = ρu⁺ - ρu⁻
    Δρe = state_prognostic⁺.energy.ρe - state_prognostic⁻.energy.ρe
    Δρq_tot =
        state_prognostic⁺.moisture.ρq_tot - state_prognostic⁻.moisture.ρq_tot
    Δstate = SVector(Δρ, Δρu[1], Δρu[2], Δρu[3], Δρe, Δρq_tot)

    parent(fluxᵀn) .-= M * Λ * (M \ Δstate) / 2
end

function numerical_flux_first_order!(
    ::LMARSNumericalFlux,
    balance_law::AtmosLinearModel,
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
        RusanovNumericalFlux(),
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
