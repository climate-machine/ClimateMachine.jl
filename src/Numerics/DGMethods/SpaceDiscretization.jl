using .NumericalFluxes:
    CentralNumericalFluxHigherOrder, CentralNumericalFluxDivergence

"""
    SpaceDiscretization

Supertype for spatial discretizations.

Must have the following properties:

    - `grid`
    - `balance_law`
    - `state_auxiliary`
"""
abstract type SpaceDiscretization end


basic_grid_info(spacedisc::SpaceDiscretization) =
    basic_grid_info(spacedisc.grid)
function basic_grid_info(grid)
    dim = dimensionality(grid)
    # Tuple of polynomial degrees (N₁, N₂, N₃)
    N = polynomialorders(grid)
    @assert dim == 2 || N[1] == N[2]

    # Number of quadrature point in each direction (Nq₁, Nq₂, Nq₃)
    Nq = N .+ 1

    # Number of quadrature point in the horizontal direction Nq₁ * Nq₂
    Nqh = Nq[1] * Nq[2]

    # Number of quadrature points in the vertical Nq₃
    Nqk = dim == 2 ? 1 : Nq[dim]

    Np = dofs_per_element(grid)
    Nfp_v, Nfp_h = div.(Np, (Nq[1], Nq[end]))

    topology_info = basic_topology_info(grid.topology)

    ninteriorelem = length(grid.interiorelems)
    nexteriorelem = length(grid.exteriorelems)

    nface = 2 * dim

    grid_info = (
        dim = dim,
        N = N,
        Nq = Nq,
        Nqh = Nqh,
        Nqk = Nqk,
        Nfp_v = Nfp_v,
        Nfp_h = Nfp_h,
        Np = Np,
        nface = nface,
        ninteriorelem = ninteriorelem,
        nexteriorelem = nexteriorelem,
    )

    return merge(grid_info, topology_info)
end

function basic_launch_info(spacedisc::SpaceDiscretization)
    device = array_device(spacedisc.state_auxiliary)
    grid_info = basic_grid_info(spacedisc.grid)
    return merge(grid_info, (device = device,))
end

function (spacedisc::SpaceDiscretization)(
    tendency,
    state_prognostic,
    param,
    t;
    increment = false,
)
    # TODO deprecate increment argument
    spacedisc(tendency, state_prognostic, param, t, true, increment)
end

function init_ode_state(
    spacedisc::SpaceDiscretization,
    args...;
    init_on_cpu = false,
    fill_nan = false,
)
    grid = spacedisc.grid
    balance_law = spacedisc.balance_law
    state_auxiliary = spacedisc.state_auxiliary

    device = arraytype(grid) <: Array ? CPU() : CUDADevice()

    state_prognostic =
        create_state(balance_law, grid, Prognostic(), fill_nan = fill_nan)

    topology = grid.topology
    Np = dofs_per_element(grid)

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    nrealelem = length(topology.realelems)

    if !init_on_cpu
        event = Event(device)
        event = kernel_init_state_prognostic!(device, min(Np, 1024))(
            balance_law,
            Val(dim),
            Val(N),
            state_prognostic.data,
            state_auxiliary.data,
            grid.vgeo,
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
            dependencies = (event,),
        )
        wait(device, event)
    else
        h_state_prognostic = similar(state_prognostic, Array)
        h_state_auxiliary = similar(state_auxiliary, Array)
        h_state_auxiliary .= state_auxiliary
        event = kernel_init_state_prognostic!(CPU(), Np)(
            balance_law,
            Val(dim),
            Val(N),
            h_state_prognostic.data,
            h_state_auxiliary.data,
            Array(grid.vgeo),
            topology.realelems,
            args...;
            ndrange = Np * nrealelem,
        )
        wait(event)
        state_prognostic .= h_state_prognostic
        state_auxiliary .= h_state_auxiliary
    end

    event = Event(device)
    event = MPIStateArrays.begin_ghost_exchange!(
        state_prognostic;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        state_prognostic;
        dependencies = event,
    )
    wait(device, event)

    return state_prognostic
end

update_auxiliary_state_gradient!(::SpaceDiscretization, _...) = false

# By default, we call update_auxiliary_state!, given
# nodal_update_auxiliary_state!, defined for the
# particular balance_law:

# TODO: this should really be a separate function
function update_auxiliary_state!(
    f!,
    spacedisc::SpaceDiscretization,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange = spacedisc.grid.topology.realelems;
    diffusive = false,
)
    device = array_device(state_prognostic)

    grid = spacedisc.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    nelem = length(elems)

    Np = dofs_per_element(grid)

    knl_nodal_update_auxiliary_state! =
        kernel_nodal_update_auxiliary_state!(device, min(Np, 1024))
    ### Update state_auxiliary variables
    event = Event(device)
    if diffusive
        event = knl_nodal_update_auxiliary_state!(
            m,
            Val(dim),
            Val(N),
            f!,
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            spacedisc.state_gradient_flux.data,
            t,
            elems,
            grid.activedofs;
            ndrange = Np * nelem,
            dependencies = (event,),
        )
    else
        event = knl_nodal_update_auxiliary_state!(
            m,
            Val(dim),
            Val(N),
            f!,
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            t,
            elems,
            grid.activedofs;
            ndrange = Np * nelem,
            dependencies = (event,),
        )
    end
    checked_wait(device, event, nothing, spacedisc.check_for_crashes)
end

function MPIStateArrays.MPIStateArray(dg::SpaceDiscretization)
    balance_law = dg.balance_law
    grid = dg.grid

    state_prognostic = create_state(balance_law, grid, Prognostic())

    return state_prognostic
end

function restart_auxiliary_state(bl, grid, aux_data, direction)
    state_auxiliary = create_state(bl, grid, Auxiliary())
    state_auxiliary =
        init_state(state_auxiliary, bl, grid, direction, Auxiliary())
    state_auxiliary .= aux_data
    return state_auxiliary
end

# TODO: this should really be a separate function
"""
    init_state_auxiliary!(
        bl::BalanceLaw,
        f!,
        statearray_auxiliary,
        grid,
        direction;
        state_temporary = nothing
    )

Apply `f!(bl, state_auxiliary, tmp, geom)` at each node, storing the result in
`statearray_auxiliary`, where `tmp` are the values at the corresponding node in
`state_temporary` and `geom` contains the geometry information.
"""
function init_state_auxiliary!(
    balance_law,
    init_f!,
    state_auxiliary,
    grid,
    direction;
    state_temporary = nothing,
)
    topology = grid.topology
    dim = dimensionality(grid)
    Np = dofs_per_element(grid)
    N = polynomialorders(grid)
    vgeo = grid.vgeo
    device = array_device(state_auxiliary)
    nrealelem = length(topology.realelems)

    event = Event(device)
    event = kernel_nodal_init_state_auxiliary!(
        device,
        min(Np, 1024),
        Np * nrealelem,
    )(
        balance_law,
        Val(dim),
        Val(N),
        init_f!,
        state_auxiliary.data,
        isnothing(state_temporary) ? nothing : state_temporary.data,
        Val(isnothing(state_temporary) ? @vars() : vars(state_temporary)),
        vgeo,
        topology.realelems,
        dependencies = (event,),
    )

    event = MPIStateArrays.begin_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    event = MPIStateArrays.end_ghost_exchange!(
        state_auxiliary;
        dependencies = event,
    )
    wait(device, event)
end

"""
    courant(local_courant::Function, dg::DGModel, m::BalanceLaw,
            state_prognostic::MPIStateArray, direction=EveryDirection())
Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain.  The function `local_courant` is given an
approximation of the local node distance `Δx`.  The `direction` controls which
reference directions are considered when computing the minimum node distance
`Δx`.
An example `local_courant` function is
    function local_courant(m::AtmosModel, state_prognostic::Vars, state_auxiliary::Vars,
                           diffusive::Vars, Δx)
      return Δt * cmax / Δx
    end
where `Δt` is the time step size and `cmax` is the maximum flow speed in the
model.
"""
function courant(
    local_courant::Function,
    dg::SpaceDiscretization,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    Δt,
    simtime,
    direction = EveryDirection(),
)
    grid = dg.grid
    topology = grid.topology
    nrealelem = length(topology.realelems)

    if nrealelem > 0
        N = polynomialorders(grid)
        dim = dimensionality(grid)
        Nq = N .+ 1
        Nqk = dim == 2 ? 1 : Nq[dim]
        device = array_device(grid.vgeo)
        pointwise_courant = similar(grid.vgeo, prod(Nq), nrealelem)
        event = Event(device)
        event = Grids.kernel_min_neighbor_distance!(
            device,
            min(Nq[1] * Nq[2] * Nqk, 1024),
        )(
            Val(N),
            Val(dim),
            direction,
            pointwise_courant,
            grid.vgeo,
            topology.realelems;
            ndrange = (nrealelem * Nq[1] * Nq[2] * Nqk),
            dependencies = (event,),
        )
        event = kernel_local_courant!(device, min(Nq[1] * Nq[2] * Nqk, 1024))(
            m,
            Val(dim),
            Val(N),
            pointwise_courant,
            local_courant,
            state_prognostic.data,
            dg.state_auxiliary.data,
            dg.state_gradient_flux.data,
            topology.realelems,
            Δt,
            simtime,
            direction;
            ndrange = (nrealelem * Nq[1] * Nq[2] * Nqk),
            dependencies = (event,),
        )
        checked_wait(device, event, nothing, dg.check_for_crashes)

        rank_courant_max = maximum(pointwise_courant)
    else
        rank_courant_max = typemin(eltype(state_prognostic))
    end

    MPI.Allreduce(rank_courant_max, max, topology.mpicomm)
end

"""
    auxiliary_field_gradient!(::BalanceLaw, ∇state::MPIStateArray,
                               vars_out, state::MPIStateArray, vars_in, grid;
                               direction = EveryDirection())

Take the gradient of the variables `vars_in` located in the array `state`
and stores it in the variables `vars_out` of `∇state`. This function computes
element wise gradient without accounting for numerical fluxes and hence
its primary purpose is to take the gradient of continuous reference fields.

## Examples
```julia
FT = eltype(state_auxiliary)
grad_Φ = similar(state_auxiliary, vars=@vars(∇Φ::SVector{3, FT}))
auxiliary_field_gradient!(
    model,
    grad_Φ,
    ("∇Φ",),
    state_auxiliary,
    ("orientation.Φ",),
    grid,
)
```
"""
function auxiliary_field_gradient!(
    m::BalanceLaw,
    ∇state::MPIStateArray,
    vars_out,
    state::MPIStateArray,
    vars_in,
    grid,
    direction = EveryDirection(),
)
    topology = grid.topology
    nrealelem = length(topology.realelems)
    N = polynomialorders(grid)
    dim = dimensionality(grid)
    Nq = N .+ 1
    Nqk = dim == 2 ? 1 : Nq[dim]
    device = array_device(state)

    I = varsindices(vars(state), vars_in)
    O = varsindices(vars(∇state), vars_out)

    event = Event(device)

    if direction isa EveryDirection || direction isa HorizontalDirection
        # We assume N₁ = N₂, so the same polyorder, quadrature weights,
        # and differentiation operators are used
        horizontal_polyorder = N[1]
        horizontal_D = grid.D[1]
        horizontal_ω = grid.ω[1]
        event = dgsem_auxiliary_field_gradient!(device, (Nq[1], Nq[2]))(
            m,
            Val(dim),
            Val(N),
            HorizontalDirection(),
            ∇state.data,
            state.data,
            grid.vgeo,
            horizontal_D,
            horizontal_ω,
            Val(I),
            Val(O),
            false,
            ndrange = (nrealelem * Nq[1], Nq[2]),
            dependencies = (event,),
        )
    end

    if direction isa EveryDirection || direction isa VerticalDirection
        vertical_polyorder = N[dim]
        if vertical_polyorder > 0
            vertical_D = grid.D[dim]
            vertical_ω = grid.ω[dim]
            event = dgsem_auxiliary_field_gradient!(device, (Nq[1], Nq[2]))(
                m,
                Val(dim),
                Val(N),
                VerticalDirection(),
                ∇state.data,
                state.data,
                grid.vgeo,
                vertical_D,
                vertical_ω,
                Val(I),
                Val(O),
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                (direction isa EveryDirection);
                ndrange = (nrealelem * Nq[1], Nq[2]),
                dependencies = (event,),
            )
        else
            info = basic_grid_info(grid)
            event = vert_fvm_auxiliary_field_gradient!(device, info.Nfp_h)(
                m,
                Val(info),
                ∇state.data,
                state.data,
                grid.vgeo,
                grid.sgeo,
                grid.vmap⁻,
                grid.vmap⁺,
                grid.elemtobndy,
                Val(I),
                Val(O),
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                (direction isa EveryDirection);
                ndrange = (nrealelem * info.Nfp_h),
                dependencies = (event,),
            )
        end
    end
    wait(device, event)
end

function hyperdiff_indexmap(balance_law, ::Type{FT}) where {FT}
    ns_hyperdiff = number_states(balance_law, Hyperdiffusive())
    if ns_hyperdiff > 0
        return varsindices(
            vars_state(balance_law, Gradient(), FT),
            fieldnames(vars_state(balance_law, GradientLaplacian(), FT)),
        )
    else
        return nothing
    end
end

"""
    launch_volume_gradients!(spacedisc, state_prognostic, t; dependencies)

Launches horizontal and vertical kernels for computing the volume gradients.
"""
function launch_volume_gradients!(spacedisc, state_prognostic, t; dependencies)
    FT = eltype(state_prognostic)
    # XXX: This is until FVM with hyperdiffusion for DG is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end

    # Workgroup is determined by the number of quadrature points
    # in the horizontal direction. For each horizontal quadrature
    # point, we operate on a stack of quadrature in the vertical
    # direction. (Iteration space is in the horizontal)
    info = basic_launch_info(spacedisc)

    # We assume (in 3-D) that both x and y directions
    # are discretized using the same polynomial order, Nq[1] == Nq[2].
    # In 2-D, the workgroup spans the entire set of quadrature points:
    # Nq[1] * Nq[2]
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2])
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `spacedisc.diffusion_direction`
    if spacedisc.diffusion_direction isa EveryDirection ||
       spacedisc.diffusion_direction isa HorizontalDirection

        # We assume N₁ = N₂, so the same polyorder, quadrature weights,
        # and differentiation operators are used
        horizontal_polyorder = info.N[1]
        horizontal_D = spacedisc.grid.D[1]
        comp_stream = volume_gradients!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            HorizontalDirection(),
            state_prognostic.data,
            spacedisc.state_gradient_flux.data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            horizontal_D,
            Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
            spacedisc.grid.topology.realelems,
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Now we call the kernel corresponding to the vertical direction
    if spacedisc isa DGModel && (
        spacedisc.diffusion_direction isa EveryDirection ||
        spacedisc.diffusion_direction isa VerticalDirection
    )

        # Vertical polynomial degree and differentiation matrix
        vertical_polyorder = info.N[info.dim]
        vertical_D = spacedisc.grid.D[info.dim]
        comp_stream = volume_gradients!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            VerticalDirection(),
            state_prognostic.data,
            spacedisc.state_gradient_flux.data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            vertical_D,
            Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
            spacedisc.grid.topology.realelems,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            !(spacedisc.diffusion_direction isa VerticalDirection),
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end
    return comp_stream
end

"""
    launch_interface_gradients!(spacedisc, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical kernels for computing the interface gradients.
The argument `surface` is either `:interior` or `:exterior`, which denotes whether
we are computing interface gradients on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_gradients!(
    spacedisc,
    state_prognostic,
    t;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    # XXX: This is until FVM with DG hyperdiffusion is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end

    FT = eltype(state_prognostic)

    info = basic_launch_info(spacedisc)
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `spacedisc.diffusion_direction`
    if spacedisc.diffusion_direction isa EveryDirection ||
       spacedisc.diffusion_direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = spacedisc.grid.interiorelems
            ndrange = workgroup * info.ninteriorelem
        else
            elems = spacedisc.grid.exteriorelems
            ndrange = workgroup * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal directions)
        horizontal_polyorder = info.N[1]

        comp_stream = dgsem_interface_gradients!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            HorizontalDirection(),
            spacedisc.numerical_flux_gradient,
            state_prognostic.data,
            spacedisc.state_gradient_flux.data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            spacedisc.grid.sgeo,
            t,
            spacedisc.grid.vmap⁻,
            spacedisc.grid.vmap⁺,
            spacedisc.grid.elemtobndy,
            Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
            elems;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical interface kernel call
    if spacedisc.diffusion_direction isa EveryDirection ||
       spacedisc.diffusion_direction isa VerticalDirection

        workgroup = info.Nfp_h
        if surface === :interior
            elems = spacedisc.grid.interiorelems
            ndrange = workgroup * info.ninteriorelem
        else
            elems = spacedisc.grid.exteriorelems
            ndrange = workgroup * info.nexteriorelem
        end

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]

        if spacedisc isa DGModel
            comp_stream = dgsem_interface_gradients!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                VerticalDirection(),
                spacedisc.numerical_flux_gradient,
                state_prognostic.data,
                spacedisc.state_gradient_flux.data,
                Qhypervisc_grad_data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.vmap⁻,
                spacedisc.grid.vmap⁺,
                spacedisc.grid.elemtobndy,
                Val(hyperdiff_indexmap(spacedisc.balance_law, FT)),
                elems;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        elseif spacedisc isa DGFVModel
            # Make sure FVM in the vertical
            @assert info.N[info.dim] == 0

            # The FVM will only work on stacked grids!
            @assert isstacked(spacedisc.grid.topology)
            nvertelem = spacedisc.grid.topology.stacksize
            periodicstack = spacedisc.grid.topology.periodicstack

            # 1 thread per degree freedom per element
            comp_stream = vert_fvm_interface_gradients!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                Val(nvertelem),
                Val(periodicstack),
                VerticalDirection(),
                state_prognostic.data,
                spacedisc.state_gradient_flux.data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.elemtobndy,
                elems,
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                spacedisc.direction isa EveryDirection,
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        else
            error("unknown spatial discretization: $(typeof(spacedisc))")
        end
    end
    return comp_stream
end

"""
    launch_volume_divergence_of_gradients!(dg, state_prognostic, t; dependencies)

Launches horizontal and vertical volume kernels for computing the divergence of gradients.
"""
function launch_volume_divergence_of_gradients!(
    dg,
    state_prognostic,
    t;
    dependencies,
)
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    info = basic_launch_info(dg)
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.nrealelem * info.Nq[1], info.Nq[2])
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        # Horizontal polynomial order and differentiation matrix
        horizontal_polyorder = info.N[1]
        horizontal_D = dg.grid.D[1]

        comp_stream = volume_divergence_of_gradients!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            HorizontalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            dg.grid.vgeo,
            horizontal_D,
            dg.grid.topology.realelems;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # And now the vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        # Vertical polynomial order and differentiation matrix
        vertical_polyorder = info.N[info.dim]
        vertical_D = dg.grid.D[info.dim]

        comp_stream = volume_divergence_of_gradients!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            VerticalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            dg.grid.vgeo,
            vertical_D,
            dg.grid.topology.realelems,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            !(dg.diffusion_direction isa VerticalDirection);
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end
    return comp_stream
end

"""
    launch_interface_divergence_of_gradients!(dg, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical interface kernels for computing the divergence of gradients.
The argument `surface` is either `:interior` or `:exterior`, which denotes whether
we are computing values on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_divergence_of_gradients!(
    dg,
    state_prognostic,
    t;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    info = basic_launch_info(dg)
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_v * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_v * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal directions)
        horizontal_polyorder = info.N[1]

        comp_stream =
            interface_divergence_of_gradients!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                HorizontalDirection(),
                CentralNumericalFluxDivergence(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                dg.state_auxiliary.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                t,
                elems;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    # Vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        workgroup = info.Nfp_h
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_h * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_h * info.nexteriorelem
        end

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]

        comp_stream =
            interface_divergence_of_gradients!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                VerticalDirection(),
                CentralNumericalFluxDivergence(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                dg.state_auxiliary.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                t,
                elems;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    return comp_stream
end

"""
    launch_volume_gradients_of_laplacians!(dg, state_prognostic, t; dependencies)

Launches horizontal and vertical volume kernels for computing the DG gradient of
a second-order DG gradient (Laplacian).
"""
function launch_volume_gradients_of_laplacians!(
    dg,
    state_prognostic,
    t;
    dependencies,
)
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order

    info = basic_launch_info(dg)
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.nrealelem * info.Nq[1], info.Nq[2])
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        # Horizontal polynomial degree
        horizontal_polyorder = info.N[1]
        # Horizontal quadrature weights and differentiation matrix
        horizontal_ω = dg.grid.ω[1]
        horizontal_D = dg.grid.D[1]

        comp_stream = volume_gradients_of_laplacians!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            HorizontalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            state_prognostic.data,
            dg.state_auxiliary.data,
            dg.grid.vgeo,
            horizontal_ω,
            horizontal_D,
            dg.grid.topology.realelems,
            t;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]
        # Vertical quadrature weights and differentiation matrix
        vertical_ω = dg.grid.ω[info.dim]
        vertical_D = dg.grid.D[info.dim]

        comp_stream = volume_gradients_of_laplacians!(info.device, workgroup)(
            dg.balance_law,
            Val(info),
            VerticalDirection(),
            Qhypervisc_grad.data,
            Qhypervisc_div.data,
            state_prognostic.data,
            dg.state_auxiliary.data,
            dg.grid.vgeo,
            vertical_ω,
            vertical_D,
            dg.grid.topology.realelems,
            t,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            !(dg.diffusion_direction isa VerticalDirection);
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    return comp_stream
end

"""
    launch_interface_gradients_of_laplacians!(dg, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical interface kernels for computing the gradients of Laplacians
(second-order gradients). The argument `surface` is either `:interior` or `:exterior`,
which denotes whether we are computing values on boundaries which are interior (exterior resp.)
to the _parallel_ boundary.
"""
function launch_interface_gradients_of_laplacians!(
    dg,
    state_prognostic,
    t;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    Qhypervisc_grad, Qhypervisc_div = dg.states_higher_order
    comp_stream = dependencies
    info = basic_launch_info(dg)

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction `dg.diffusion_direction`
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_v * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_v * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal directions)
        horizontal_polyorder = info.N[1]

        comp_stream =
            interface_gradients_of_laplacians!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                HorizontalDirection(),
                CentralNumericalFluxHigherOrder(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                state_prognostic.data,
                dg.state_auxiliary.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                elems,
                t;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    # Vertical kernel call
    if dg.diffusion_direction isa EveryDirection ||
       dg.diffusion_direction isa VerticalDirection

        workgroup = info.Nfp_h
        if surface === :interior
            elems = dg.grid.interiorelems
            ndrange = info.Nfp_h * info.ninteriorelem
        else
            elems = dg.grid.exteriorelems
            ndrange = info.Nfp_h * info.nexteriorelem
        end

        # Vertical polynomial degree
        vertical_polyorder = info.N[info.dim]

        comp_stream =
            interface_gradients_of_laplacians!(info.device, workgroup)(
                dg.balance_law,
                Val(info),
                VerticalDirection(),
                CentralNumericalFluxHigherOrder(),
                Qhypervisc_grad.data,
                Qhypervisc_div.data,
                state_prognostic.data,
                dg.state_auxiliary.data,
                dg.grid.vgeo,
                dg.grid.sgeo,
                dg.grid.vmap⁻,
                dg.grid.vmap⁺,
                dg.grid.elemtobndy,
                elems,
                t;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
    end

    return comp_stream
end

"""
    launch_volume_tendency!(spacedisc, state_prognostic, t; dependencies)

Launches horizontal and vertical volume kernels for computing tendencies (sources, sinks, etc).
"""
function launch_volume_tendency!(
    spacedisc,
    tendency,
    state_prognostic,
    t,
    α,
    β;
    dependencies,
)
    # XXX: This is until FVM with hyperdiffusion is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end
    grad_flux_data = spacedisc.state_gradient_flux.data

    # Workgroup is determined by the number of quadrature points
    # in the horizontal direction. For each horizontal quadrature
    # point, we operate on a stack of quadrature in the vertical
    # direction. (Iteration space is in the horizontal)
    info = basic_launch_info(spacedisc)

    # We assume (in 3-D) that both x and y directions
    # are discretized using the same polynomial order, Nq[1] == Nq[2].
    # In 2-D, the workgroup spans the entire set of quadrature points:
    # Nq[1] * Nq[2]
    workgroup = (info.Nq[1], info.Nq[2])
    ndrange = (info.Nq[1] * info.nrealelem, info.Nq[2])
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

        comp_stream = volume_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            spacedisc.direction,
            HorizontalDirection(),
            tendency.data,
            state_prognostic.data,
            grad_flux_data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            horizontal_ω,
            horizontal_D,
            spacedisc.grid.topology.realelems,
            α,
            β,
            # If the model direction is horizontal or FV in the vertical,
            # we want to be sure to add sources
            spacedisc.direction isa HorizontalDirection ||
            spacedisc isa DGFVModel,
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
        # Vertical quadrature weights and differentiation matrix
        vertical_ω = spacedisc.grid.ω[info.dim]
        vertical_D = spacedisc.grid.D[info.dim]

        comp_stream = volume_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            spacedisc.direction,
            VerticalDirection(),
            tendency.data,
            state_prognostic.data,
            grad_flux_data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            t,
            vertical_ω,
            vertical_D,
            spacedisc.grid.topology.realelems,
            α,
            # If we are computing the volume gradient in every direction, we
            # need to increment into the appropriate fields _after_ the
            # horizontal computation.
            spacedisc.direction isa EveryDirection ? true : β,
            # Boolean to add source. In the case of EveryDirection, we always add the sources
            # in the vertical kernel. Here, we make the assumption that we're either computing
            # in every direction, or _just_ the vertical direction.
            true;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    return comp_stream
end

"""
    launch_interface_tendency!(spacedisc, state_prognostic, t; surface::Symbol, dependencies)

Launches horizontal and vertical interface kernels for computing tendencies (sources, sinks, etc).
The argument `surface` is either `:interior` or `:exterior`, which denotes whether we are computing
values on boundaries which are interior (exterior resp.) to the _parallel_ boundary.
"""
function launch_interface_tendency!(
    spacedisc,
    tendency,
    state_prognostic,
    t,
    α,
    β;
    surface::Symbol,
    dependencies,
)
    @assert surface === :interior || surface === :exterior
    # XXX: This is until FVM with diffusion is implemented
    if spacedisc isa DGFVModel
        @assert 0 == number_states(spacedisc.balance_law, Hyperdiffusive())
        Qhypervisc_grad_data = nothing
    elseif spacedisc isa DGModel
        Qhypervisc_grad_data = spacedisc.states_higher_order[1].data
    end
    grad_flux_data = spacedisc.state_gradient_flux.data
    numerical_flux_second_order = spacedisc.numerical_flux_second_order

    info = basic_launch_info(spacedisc)
    comp_stream = dependencies

    # If the model direction is EveryDirection, we need to perform
    # both horizontal AND vertical kernel calls; otherwise, we only
    # call the kernel corresponding to the model direction
    # `spacedisc.diffusion_direction`
    if spacedisc.direction isa EveryDirection ||
       spacedisc.direction isa HorizontalDirection

        workgroup = info.Nfp_v
        if surface === :interior
            elems = spacedisc.grid.interiorelems
            ndrange = workgroup * info.ninteriorelem
        else
            elems = spacedisc.grid.exteriorelems
            ndrange = workgroup * info.nexteriorelem
        end

        # Hoirzontal polynomial order (assumes same for both horizontal
        # directions)
        horizontal_polyorder = info.N[1]

        comp_stream = dgsem_interface_tendency!(info.device, workgroup)(
            spacedisc.balance_law,
            Val(info),
            HorizontalDirection(),
            spacedisc.numerical_flux_first_order,
            numerical_flux_second_order,
            tendency.data,
            state_prognostic.data,
            grad_flux_data,
            Qhypervisc_grad_data,
            spacedisc.state_auxiliary.data,
            spacedisc.grid.vgeo,
            spacedisc.grid.sgeo,
            t,
            spacedisc.grid.vmap⁻,
            spacedisc.grid.vmap⁺,
            spacedisc.grid.elemtobndy,
            elems,
            α;
            ndrange = ndrange,
            dependencies = comp_stream,
        )
    end

    # Vertical kernel call
    if spacedisc.direction isa EveryDirection ||
       spacedisc.direction isa VerticalDirection
        elems =
            surface === :interior ? elems = spacedisc.grid.interiorelems :
            spacedisc.grid.exteriorelems

        if spacedisc isa DGModel
            workgroup = info.Nfp_h
            ndrange = workgroup * length(elems)

            # Vertical polynomial degree
            vertical_polyorder = info.N[info.dim]

            comp_stream = dgsem_interface_tendency!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                VerticalDirection(),
                spacedisc.numerical_flux_first_order,
                numerical_flux_second_order,
                tendency.data,
                state_prognostic.data,
                grad_flux_data,
                Qhypervisc_grad_data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.vmap⁻,
                spacedisc.grid.vmap⁺,
                spacedisc.grid.elemtobndy,
                elems,
                α;
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        elseif spacedisc isa DGFVModel
            # Make sure FVM in the vertical
            @assert info.N[info.dim] == 0

            # The FVM will only work on stacked grids!
            @assert isstacked(spacedisc.grid.topology)

            # Figute out the stacking of the mesh
            nvertelem = spacedisc.grid.topology.stacksize
            nhorzelem = div(length(elems), nvertelem)
            periodicstack = spacedisc.grid.topology.periodicstack

            # 2-D workgroup
            workgroup = info.Nfp_h
            ndrange = workgroup * nhorzelem

            # XXX: This will need to be updated to diffusion
            comp_stream = vert_fvm_interface_tendency!(info.device, workgroup)(
                spacedisc.balance_law,
                Val(info),
                Val(nvertelem),
                Val(periodicstack),
                VerticalDirection(),
                spacedisc.fv_reconstruction,
                spacedisc.numerical_flux_first_order,
                numerical_flux_second_order,
                tendency.data,
                state_prognostic.data,
                grad_flux_data,
                spacedisc.state_auxiliary.data,
                spacedisc.grid.vgeo,
                spacedisc.grid.sgeo,
                t,
                spacedisc.grid.elemtobndy,
                elems,
                α,
                β,
                # If we are computing in every direction, we need to
                # increment after we compute the horizontal values
                spacedisc.direction isa EveryDirection,
                # If we are computing in vertical direction, we need to
                # add sources here
                spacedisc.direction isa VerticalDirection,
                ndrange = ndrange,
                dependencies = comp_stream,
            )
        else
            error("unknown spatial discretization: $(typeof(spacedisc))")
        end
    end

    return comp_stream
end
