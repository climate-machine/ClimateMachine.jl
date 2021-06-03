module DGMethods

using MPI
using StaticArrays
using DocStringExtensions
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using ..MPIStateArrays
using ..Mesh.Grids
using ..Mesh.Topologies
using ..Mesh.Filters
using ..VariableTemplates
using ..Courant
using ..BalanceLaws:
    BalanceLaw,
    AbstractStateType,
    Prognostic,
    Auxiliary,
    Gradient,
    GradientFlux,
    GradientLaplacian,
    Hyperdiffusive,
    UpwardIntegrals,
    DownwardIntegrals,
    vars_state,
    number_states

import ..BalanceLaws:
    BalanceLaw,
    init_state_prognostic_arr!,
    init_state_auxiliary!,
    flux_first_order_arr!,
    total_flux_first_order_arr!,
    two_point_flux_first_order_arr!,
    flux_second_order_arr!,
    compute_gradient_flux_arr!,
    compute_gradient_argument_arr!,
    source_arr!,
    transform_post_gradient_laplacian_arr!,
    wavespeed,
    boundary_conditions,
    boundary_state!,
    update_auxiliary_state!,
    update_auxiliary_state_gradient!,
    nodal_update_auxiliary_state!,
    nodal_init_state_auxiliary!,
    integral_load_auxiliary_state_arr!,
    integral_set_auxiliary_state_arr!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    reverse_integral_load_auxiliary_state_arr!,
    reverse_integral_set_auxiliary_state_arr!

export DGModel,
    DGFVModel,
    ESDGModel,
    SpaceDiscretization,
    init_ode_state,
    restart_ode_state,
    restart_auxiliary_state,
    basic_grid_info,
    init_state_auxiliary!,
    auxiliary_field_gradient!,
    courant

include("custom_filter.jl")
include("FVReconstructions.jl")
include("NumericalFluxes.jl")
include("SpaceDiscretization.jl")
include("DGModel.jl")
include("DGFVModel.jl")
include("ESDGModel.jl")
include("create_states.jl")

"""
    calculate_dt(dg, model, Q, Courant_number, direction, t)

For a given model, compute a time step satisying the nondiffusive Courant number
`Courant_number`
"""
function calculate_dt(dg, model, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = courant(nondiffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end

end
