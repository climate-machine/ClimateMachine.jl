include("DGModel_kernels.jl")

struct DGModel{BL, G, NFND, NFD, GNF, AS, DS, HDS, D, DD, MD, GF, TF} <:
       SpaceDiscretization
    balance_law::BL
    grid::G
    numerical_flux_first_order::NFND
    numerical_flux_second_order::NFD
    numerical_flux_gradient::GNF
    state_auxiliary::AS
    state_gradient_flux::DS
    states_higher_order::HDS
    direction::D
    diffusion_direction::DD
    modeldata::MD
    gradient_filter::GF
    tendency_filter::TF
    check_for_crashes::Bool
end


function DGModel(
    balance_law,
    grid,
    numerical_flux_first_order,
    numerical_flux_second_order,
    numerical_flux_gradient;
    fill_nan = false,
    check_for_crashes = false,
    state_auxiliary = create_state(
        balance_law,
        grid,
        Auxiliary(),
        fill_nan = fill_nan,
    ),
    state_gradient_flux = create_state(balance_law, grid, GradientFlux()),
    states_higher_order = (
        create_state(balance_law, grid, GradientLaplacian()),
        create_state(balance_law, grid, Hyperdiffusive()),
    ),
    direction = EveryDirection(),
    diffusion_direction = direction,
    modeldata = nothing,
    gradient_filter = nothing,
    tendency_filter = nothing,
)
    state_auxiliary =
        init_state(state_auxiliary, balance_law, grid, direction, Auxiliary())
    DGModel(
        balance_law,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        state_auxiliary,
        state_gradient_flux,
        states_higher_order,
        direction,
        diffusion_direction,
        modeldata,
        gradient_filter,
        tendency_filter,
        check_for_crashes,
    )
end

# Include the remainder model for composing DG models and balance laws
include("remainder.jl")

"""
    (dg::DGModel)(tendency, state_prognostic, _, t, α, β)

Uses spectral element discontinuous Galerkin in all direction to compute the
tendency.

Computes the tendency terms compatible with `IncrementODEProblem`

    tendency .= α .* dQdt(state_prognostic, p, t) .+ β .* tendency

The 4-argument form will just compute

    tendency .= dQdt(state_prognostic, p, t)
"""
function (dg::DGModel)(tendency, state_prognostic, _, t, α, β)

    device = array_device(state_prognostic)
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(dg.balance_law, Prognostic())
    num_state_gradient_flux = number_states(dg.balance_law, GradientFlux())
    nhyperviscstate = number_states(dg.balance_law, Hyperdiffusive())
    num_state_tendency = size(tendency, 2)

    @assert num_state_prognostic ≤ num_state_tendency

    if num_state_prognostic < num_state_tendency && β != 1
        # If we don't operate on the full state, then we need to scale here instead of volume_tendency!
        tendency .*= β
        β = β != 0 # if β==0 then we can avoid the memory load in volume_tendency!
    end

    communicate =
        !(
            isstacked(dg.grid.topology) &&
            typeof(dg.direction) <: VerticalDirection
        )

    update_auxiliary_state!(
        dg,
        dg.balance_law,
        state_prognostic,
        t,
        dg.grid.topology.realelems,
    )

    exchange_state_prognostic = NoneEvent()
    exchange_state_gradient_flux = NoneEvent()
    exchange_Qhypervisc_grad = NoneEvent()
    exchange_Qhypervisc_div = NoneEvent()

    comp_stream = Event(device)

    if communicate
        exchange_state_prognostic = MPIStateArrays.begin_ghost_exchange!(
            state_prognostic;
            dependencies = comp_stream,
        )
    end

    if num_state_gradient_flux > 0 || nhyperviscstate > 0
        ########################
        # Gradient Computation #
        ########################

        comp_stream = launch_volume_gradients!(
            dg,
            state_prognostic,
            t;
            dependencies = comp_stream,
        )

        comp_stream = launch_interface_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :interior,
            dependencies = comp_stream,
        )

        if communicate
            exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
                state_prognostic;
                dependencies = exchange_state_prognostic,
                check_for_crashes = dg.check_for_crashes,
            )

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            checked_wait(
                device,
                exchange_state_prognostic,
                nothing,
                dg.check_for_crashes,
            )
            update_auxiliary_state!(
                dg,
                dg.balance_law,
                state_prognostic,
                t,
                dg.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
        end

        comp_stream = launch_interface_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :exterior,
            dependencies = (comp_stream, exchange_state_prognostic),
        )

        if dg.gradient_filter !== nothing
            comp_stream = Filters.apply_async!(
                dg.state_gradient_flux,
                1:num_state_gradient_flux,
                dg.grid,
                dg.gradient_filter;
                dependencies = comp_stream,
            )
        end

        if communicate
            if num_state_gradient_flux > 0
                exchange_state_gradient_flux =
                    MPIStateArrays.begin_ghost_exchange!(
                        dg.state_gradient_flux,
                        dependencies = comp_stream,
                    )
            end
            if nhyperviscstate > 0
                exchange_Qhypervisc_grad = MPIStateArrays.begin_ghost_exchange!(
                    Qhypervisc_grad,
                    dependencies = comp_stream,
                )
            end
        end

        if num_state_gradient_flux > 0
            # Update_aux_diffusive may start asynchronous work on the compute device
            # and we synchronize those here through a device event.
            checked_wait(device, comp_stream, nothing, dg.check_for_crashes)
            update_auxiliary_state_gradient!(
                dg,
                dg.balance_law,
                state_prognostic,
                t,
                dg.grid.topology.realelems,
            )
            comp_stream = Event(device)
        end
    end

    if nhyperviscstate > 0
        #########################
        # Laplacian Computation #
        #########################

        comp_stream = launch_volume_divergence_of_gradients!(
            dg,
            state_prognostic,
            t;
            dependencies = comp_stream,
        )

        comp_stream = launch_interface_divergence_of_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :interior,
            dependencies = comp_stream,
        )

        if communicate
            exchange_Qhypervisc_grad = MPIStateArrays.end_ghost_exchange!(
                Qhypervisc_grad,
                dependencies = exchange_Qhypervisc_grad,
                check_for_crashes = dg.check_for_crashes,
            )
        end

        comp_stream = launch_interface_divergence_of_gradients!(
            dg,
            state_prognostic,
            t;
            surface = :exterior,
            dependencies = (comp_stream, exchange_Qhypervisc_grad),
        )

        if communicate
            exchange_Qhypervisc_div = MPIStateArrays.begin_ghost_exchange!(
                Qhypervisc_div,
                dependencies = comp_stream,
            )
        end

        ####################################
        # Hyperdiffusive terms computation #
        ####################################

        comp_stream = launch_volume_gradients_of_laplacians!(
            dg,
            state_prognostic,
            t,
            dependencies = (comp_stream,),
        )

        comp_stream = launch_interface_gradients_of_laplacians!(
            dg,
            state_prognostic,
            t;
            surface = :interior,
            dependencies = (comp_stream,),
        )

        if communicate
            exchange_Qhypervisc_div = MPIStateArrays.end_ghost_exchange!(
                Qhypervisc_div,
                dependencies = exchange_Qhypervisc_div,
                check_for_crashes = dg.check_for_crashes,
            )
        end

        comp_stream = launch_interface_gradients_of_laplacians!(
            dg,
            state_prognostic,
            t;
            surface = :exterior,
            dependencies = (comp_stream, exchange_Qhypervisc_div),
        )

        if communicate
            exchange_Qhypervisc_grad = MPIStateArrays.begin_ghost_exchange!(
                Qhypervisc_grad,
                dependencies = comp_stream,
            )
        end
    end


    ###################
    # RHS Computation #
    ###################
    comp_stream = launch_volume_tendency!(
        dg,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        dependencies = (comp_stream,),
    )

    comp_stream = launch_fluxdiff_volume_tendency!(
        dg,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        dependencies = (comp_stream,),
    )

    comp_stream = launch_interface_tendency!(
        dg,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        surface = :interior,
        dependencies = (comp_stream,),
    )

    if communicate
        if num_state_gradient_flux > 0 || nhyperviscstate > 0
            if num_state_gradient_flux > 0
                exchange_state_gradient_flux =
                    MPIStateArrays.end_ghost_exchange!(
                        dg.state_gradient_flux;
                        dependencies = exchange_state_gradient_flux,
                        check_for_crashes = dg.check_for_crashes,
                    )

                # Update_aux_diffusive may start asynchronous work on the
                # compute device and we synchronize those here through a device
                # event.
                checked_wait(
                    device,
                    exchange_state_gradient_flux,
                    nothing,
                    dg.check_for_crashes,
                )
                update_auxiliary_state_gradient!(
                    dg,
                    dg.balance_law,
                    state_prognostic,
                    t,
                    dg.grid.topology.ghostelems,
                )
                exchange_state_gradient_flux = Event(device)
            end
            if nhyperviscstate > 0
                exchange_Qhypervisc_grad = MPIStateArrays.end_ghost_exchange!(
                    Qhypervisc_grad;
                    dependencies = exchange_Qhypervisc_grad,
                    check_for_crashes = dg.check_for_crashes,
                )
            end
        else
            exchange_state_prognostic = MPIStateArrays.end_ghost_exchange!(
                state_prognostic;
                dependencies = exchange_state_prognostic,
                check_for_crashes = dg.check_for_crashes,
            )

            # Update_aux may start asynchronous work on the compute device and
            # we synchronize those here through a device event.
            checked_wait(
                device,
                exchange_state_prognostic,
                nothing,
                dg.check_for_crashes,
            )
            update_auxiliary_state!(
                dg,
                dg.balance_law,
                state_prognostic,
                t,
                dg.grid.topology.ghostelems,
            )
            exchange_state_prognostic = Event(device)
        end
    end

    comp_stream = launch_interface_tendency!(
        dg,
        tendency,
        state_prognostic,
        t,
        α,
        β;
        surface = :exterior,
        dependencies = (
            comp_stream,
            exchange_state_prognostic,
            exchange_state_gradient_flux,
            exchange_Qhypervisc_grad,
        ),
    )

    # The synchronization here through a device event prevents CuArray based and
    # other default stream kernels from launching before the work scheduled in
    # this function is finished.
    if dg.tendency_filter !== nothing
        comp_stream = Filters.apply_async!(
            tendency,
            1:num_state_tendency,
            dg.grid,
            dg.tendency_filter;
            dependencies = comp_stream,
        )
    end
    checked_wait(device, comp_stream, nothing, dg.check_for_crashes)
end

function restart_ode_state(dg::DGModel, state_data; init_on_cpu = false)
    bl = dg.balance_law
    grid = dg.grid

    state = create_state(bl, grid, Prognostic())
    state .= state_data

    device = arraytype(dg.grid) <: Array ? CPU() : CUDADevice()
    event = Event(device)
    event = MPIStateArrays.begin_ghost_exchange!(state; dependencies = event)
    event = MPIStateArrays.end_ghost_exchange!(state; dependencies = event)
    wait(device, event)

    return state
end

function indefinite_stack_integral!(
    dg::DGModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    state_auxiliary::MPIStateArray,
    t::Real,
    elems::UnitRange = dg.grid.topology.elems,
)

    device = array_device(state_prognostic)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    Nq = N .+ 1
    Nqj = dim == 2 ? 1 : Nq[2]

    FT = eltype(state_prognostic)

    # Compute integrals
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    event = Event(device)
    event = kernel_indefinite_stack_integral!(device, (Nq[1], Nqj))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        state_prognostic.data,
        state_auxiliary.data,
        grid.vgeo,
        # Only need the vertical Imat since this kernel is vertically oriented
        grid.Imat[dim],
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
        dependencies = (event,),
    )
    checked_wait(device, event, nothing, dg.check_for_crashes)
end

function reverse_indefinite_stack_integral!(
    dg::DGModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    state_auxiliary::MPIStateArray,
    t::Real,
    elems::UnitRange = dg.grid.topology.elems,
)

    device = array_device(state_auxiliary)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    Nq = N .+ 1
    Nqj = dim == 2 ? 1 : Nq[2]

    FT = eltype(state_auxiliary)

    # Compute integrals
    nelem = length(elems)
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    event = Event(device)
    event = kernel_reverse_indefinite_stack_integral!(device, (Nq[1], Nqj))(
        m,
        Val(dim),
        Val(N),
        Val(nvertelem),
        state_prognostic.data,
        state_auxiliary.data,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
        dependencies = (event,),
    )
    checked_wait(device, event, nothing, dg.check_for_crashes)
end

"""
    launch_fluxdiff_volume_tendency!(spacedisc, state_prognostic, t; dependencies)

Launches horizontal and vertical volume kernels for computing tendencies using flux differencing.
"""
function launch_fluxdiff_volume_tendency!(
    spacedisc,
    tendency,
    state_prognostic,
    t,
    α,
    β;
    dependencies,
)
    # Workgroup is determined by the number of quadrature points
    # in the horizontal direction. For each horizontal quadrature
    # point, we operate on a stack of quadrature in the vertical
    # direction. (Iteration space is in the horizontal)
    info = basic_launch_info(spacedisc)

    # We assume (in 3-D) that both x and y directions
    # are discretized using the same polynomial order, Nq[1] == Nq[2].
    workgroup = (info.Nq[1], info.Nq[2], info.Nqk)
    ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2], info.Nqk)
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction
    # `spacedisc.diffusion_direction`
    if spacedisc.direction isa EveryDirection ||
       spacedisc.direction isa HorizontalDirection

        # Horizontal polynomial degree
        horizontal_polyorder = info.N[1]
        # Horizontal quadrature weights and differentiation matrix
        horizontal_ω = spacedisc.grid.ω[1]
        horizontal_D = spacedisc.grid.D[1]

        comp_stream = fluxdiff_volume_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(1), # ξ1
            Val(info),
            tendency.data,
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            horizontal_D,
            t,
            α,
            ndrange = ndrange,
            dependencies = comp_stream,
        )

        comp_stream = fluxdiff_volume_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(2), # ξ2
            Val(info),
            tendency.data,
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            horizontal_D,
            t,
            α,
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical kernel
    if spacedisc isa DGModel && (
        spacedisc.direction isa EveryDirection ||
        spacedisc.direction isa VerticalDirection
    )

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]
        # Vertical differentiation matrix
        vertical_D = spacedisc.grid.D[info.dim]

        comp_stream = fluxdiff_volume_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(3), # ξ3
            Val(info),
            tendency.data,
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            vertical_D,
            t,
            α,
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    return comp_stream
end
