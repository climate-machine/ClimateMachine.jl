# These functions (variants of sa_numerical_method)
# return an instance of a numerical method to solve
# saturation adjustment, for different combinations
# of thermodynamic variable inputs.

# @print only accepts literal strings, so we must
# branch to print which method is being used.
function print_numerical_method(
    ::Type{sat_adjust_method},
) where {sat_adjust_method}
    if sat_adjust_method <: NewtonsMethod
        @print("    Method=NewtonsMethod")
    elseif sat_adjust_method <: NewtonsMethodAD
        @print("    Method=NewtonsMethodAD")
    elseif sat_adjust_method <: SecantMethod
        @print("    Method=SecantMethod")
    elseif sat_adjust_method <: RegulaFalsiMethod
        @print("    Method=RegulaFalsiMethod")
    else
        error("Unsupported numerical method")
    end
end

#####
##### Thermodynamic variable inputs: ρ, e_int, q_tot
#####
function sa_numerical_method(
    ::Type{NM},
    param_set::APS,
    ρ::FT,
    e_int::FT,
    q_tot::FT,
    phase_type::Type{<:PhaseEquil},
) where {FT, NM <: NewtonsMethod}
    _T_min::FT = T_min(param_set)
    T_init =
        max(_T_min, air_temperature(param_set, e_int, PhasePartition(q_tot))) # Assume all vapor
    return NewtonsMethod(
        T_init,
        T_ -> ∂e_int_∂T(param_set, heavisided(T_), e_int, ρ, q_tot, phase_type),
    )
end

function sa_numerical_method(
    ::Type{NM},
    param_set::APS,
    ρ::FT,
    e_int::FT,
    q_tot::FT,
    phase_type::Type{<:PhaseEquil},
) where {FT, NM <: NewtonsMethodAD}
    _T_min::FT = T_min(param_set)
    T_init =
        max(_T_min, air_temperature(param_set, e_int, PhasePartition(q_tot))) # Assume all vapor
    return NewtonsMethodAD(T_init)
end

function sa_numerical_method(
    ::Type{NM},
    param_set::APS,
    ρ::FT,
    e_int::FT,
    q_tot::FT,
    phase_type::Type{<:PhaseEquil},
) where {FT, NM <: SecantMethod}
    _T_min::FT = T_min(param_set)
    q_pt = PhasePartition(q_tot, FT(0), q_tot) # Assume all ice
    T_2 = air_temperature(param_set, e_int, q_pt)
    T_1 = max(_T_min, air_temperature(param_set, e_int, PhasePartition(q_tot))) # Assume all vapor
    T_2 = bound_upper_temperature(T_1, T_2)
    return SecantMethod(T_1, T_2)
end

function sa_numerical_method(
    ::Type{NM},
    param_set::APS,
    ρ::FT,
    e_int::FT,
    q_tot::FT,
    phase_type::Type{<:PhaseEquil},
) where {FT, NM <: RegulaFalsiMethod}
    _T_min::FT = T_min(param_set)

    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    _cv_d::FT = cv_d(param_set)
    _cv_l::FT = cv_l(param_set)
    _T_0::FT = T_0(param_set)

    q_pt = PhasePartition(q_tot, FT(0), q_tot) # Assume all ice

    # Sometimes T_1 is slightly larger than the root.
    # So, we subtract a small amount to ensure the root is bounded

    T_2 = air_temperature(param_set, e_int, q_pt)
    T_1 = air_temperature(param_set, e_int, PhasePartition(q_tot)) - FT(0.1) # Assume all vapor
    T_1 = max(_T_min, T_1)

    # This bound works in the reference states, but not in the full 3D input space
    # T_2 = _T_0 + ( e_int + q_tot * _e_int_i0 )  / _cv_d + FT(0.1)
    # T_1 = _T_0 +
    #        ( e_int - q_tot * _e_int_v0 )  / (_cv_d + (_cv_l - _cv_d) * q_tot) - FT(0.1)
    # T_1 = max(_T_min, T_1)

    return RegulaFalsiMethod(T_1, T_2)
end

#####
##### Thermodynamic variable inputs: ρ, p, q_tot
#####

function sa_numerical_method_ρpq(
    ::Type{NM},
    param_set::APS,
    ρ::FT,
    p::FT,
    q_tot::FT,
    phase_type::Type{<:PhaseEquil},
) where {FT, NM <: NewtonsMethodAD}
    q_pt = PhasePartition(q_tot)
    T_init = air_temperature_from_ideal_gas_law(param_set, p, ρ, q_pt)
    return NewtonsMethodAD(T_init)
end

function sa_numerical_method_ρpq(
    ::Type{NM},
    param_set::APS,
    ρ::FT,
    p::FT,
    q_tot::FT,
    phase_type::Type{<:PhaseEquil},
) where {FT, NM <: RegulaFalsiMethod}
    q_pt = PhasePartition(q_tot)
    T_1 = air_temperature_from_ideal_gas_law(param_set, p, ρ, q_pt) - 5
    T_2 = air_temperature_from_ideal_gas_law(param_set, p, ρ, q_pt) + 5
    return RegulaFalsiMethod(T_1, T_2)
end
