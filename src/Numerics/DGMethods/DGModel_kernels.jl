using .NumericalFluxes:
    numerical_flux_gradient!,
    numerical_flux_first_order!,
    numerical_flux_second_order!,
    numerical_flux_divergence!,
    numerical_flux_higher_order!,
    numerical_boundary_flux_gradient!,
    numerical_boundary_flux_first_order!,
    numerical_boundary_flux_second_order!,
    numerical_boundary_flux_divergence!,
    numerical_boundary_flux_higher_order!,
    CentralNumericalFluxGradient

using ..Mesh.Geometry

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _x1, _x2, _x3 = Grids._x1, Grids._x2, Grids._x3
const _JcV = Grids._JcV

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI
# }}}

"""
    function volume_tendency!(
        balance_law::BalanceLaw,
        ::Val{dim},
        ::Val{polyorder},
        model_direction,
        direction,
        tendency,
        state_prognostic,
        state_gradient_flux,
        Qhypervisc_grad,
        state_auxiliary,
        vgeo,
        t,
        ω,
        D,
        elems,
        α,
        β,
        add_source = false,
    )

Compute kernel for evaluating the volume tendencies for the
DG form:

∫ₑ ψ⋅ ∂q/∂t dx - ∫ₑ ∇ψ⋅(Fⁱⁿᵛ + Fᵛⁱˢᶜ) dx + ∮ₑ n̂ ψ⋅(Fⁱⁿᵛ* + Fᵛⁱˢᶜ*) dS,

or equivalently in matrix form:

dQ/dt = M⁻¹(MS + DᵀM(Fⁱⁿᵛ + Fᵛⁱˢᶜ) + ∑ᶠ LᵀMf(Fⁱⁿᵛ* + Fᵛⁱˢᶜ*)).

This kernel computes the volume terms: M⁻¹(MS + DᵀM(Fⁱⁿᵛ + Fᵛⁱˢᶜ)),
where M is the mass matrix and D is the differentiation matrix,
S is the source, and Fⁱⁿᵛ, Fᵛⁱˢᶜ are the inviscid and viscous
fluxes, respectively.
"""
@kernel function volume_tendency!(
    balance_law::BalanceLaw,
    ::Val{info},
    model_direction,
    direction,
    tendency,
    state_prognostic,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    ω,
    D,
    elems,
    α,
    β,
    add_source = false,
) where {info}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_auxiliary = number_states(balance_law, Auxiliary())

        ngradlapstate = number_states(balance_law, GradientLaplacian())
        nhyperviscstate = number_states(balance_law, Hyperdiffusive())

        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        local_source = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_flux = MArray{Tuple{3, num_state_prognostic}, FT}(undef)
        local_flux_3 = MArray{Tuple{num_state_prognostic}, FT}(undef)
    end

    # Arrays for F, and the differentiation matrix D
    shared_flux = @localmem FT (2, Nq1, Nq2, num_state_prognostic)

    # Storage for tendency and mass inverse M⁻¹
    local_tendency = @private FT (Nq3, num_state_prognostic)
    local_MI = @private FT (Nq3,)

    # Grab the index associated with the current element `e` and the
    # horizontal quadrature indices `i` (in the ξ1-direction),
    # `j` (in the ξ2-direction) [directions on the reference element].
    # Parallelize over elements, then over columns
    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            # initialize local tendency
            @unroll for s in 1:num_state_prognostic
                local_tendency[k, s] = zero(FT)
            end
            # read in mass matrix inverse for element `e`
            local_MI[k] = vgeo[ijk, _MI, e]
        end

        @unroll for k in 1:Nq3
            @synchronize
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            M = vgeo[ijk, _M, e]

            # Extract Jacobian terms ∂ξᵢ/∂xⱼ
            ξ1x1 = vgeo[ijk, _ξ1x1, e]
            ξ1x2 = vgeo[ijk, _ξ1x2, e]
            ξ1x3 = vgeo[ijk, _ξ1x3, e]
            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                ξ2x1 = vgeo[ijk, _ξ2x1, e]
                ξ2x2 = vgeo[ijk, _ξ2x2, e]
                ξ2x3 = vgeo[ijk, _ξ2x3, e]
            end
            if dim == 3 && direction isa EveryDirection
                ξ3x1 = vgeo[ijk, _ξ3x1, e]
                ξ3x2 = vgeo[ijk, _ξ3x2, e]
                ξ3x3 = vgeo[ijk, _ξ3x3, e]
            end

            # Read fields into registers (hopefully)

            # need to interpolate here
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end

            @unroll for s in 1:num_state_gradient_flux
                local_state_gradient_flux[s] = state_gradient_flux[ijk, s, e]
            end

            @unroll for s in 1:nhyperviscstate
                local_state_hyperdiffusion[s] = Qhypervisc_grad[ijk, s, e]
            end

            # Computes the local inviscid fluxes Fⁱⁿᵛ
            fill!(local_flux, -zero(eltype(local_flux)))
            flux_first_order_arr!(
                balance_law,
                local_flux,
                local_state_prognostic,
                local_state_auxiliary,
                t,
                (model_direction,),
            )

            @unroll for s in 1:num_state_prognostic
                shared_flux[1, i, j, s] = local_flux[1, s]
                shared_flux[2, i, j, s] = local_flux[2, s]
                local_flux_3[s] = local_flux[3, s]
            end

            # Computes the local viscous fluxes Fᵛⁱˢᶜ
            fill!(local_flux, -zero(eltype(local_flux)))
            flux_second_order_arr!(
                balance_law,
                local_flux,
                local_state_prognostic,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                local_state_auxiliary,
                t,
            )

            @unroll for s in 1:num_state_prognostic
                shared_flux[1, i, j, s] += local_flux[1, s]
                shared_flux[2, i, j, s] += local_flux[2, s]
                local_flux_3[s] += local_flux[3, s]
            end

            # Build "inside metrics" flux
            @unroll for s in 1:num_state_prognostic
                F1, F2, F3 = shared_flux[1, i, j, s],
                shared_flux[2, i, j, s],
                local_flux_3[s]

                shared_flux[1, i, j, s] =
                    M * (ξ1x1 * F1 + ξ1x2 * F2 + ξ1x3 * F3)
                if dim == 3 || (dim == 2 && direction isa EveryDirection)
                    shared_flux[2, i, j, s] =
                        M * (ξ2x1 * F1 + ξ2x2 * F2 + ξ2x3 * F3)
                end
                if dim == 3 && direction isa EveryDirection
                    local_flux_3[s] = M * (ξ3x1 * F1 + ξ3x2 * F2 + ξ3x3 * F3)
                end
            end

            # In the case of the remainder model we may need to loop through the
            # models to add in restricted direction components
            if model_direction isa EveryDirection && balance_law isa RemBL
                if rembl_has_subs_direction(HorizontalDirection(), balance_law)
                    fill!(local_flux, -zero(eltype(local_flux)))
                    flux_first_order_arr!(
                        balance_law,
                        local_flux,
                        local_state_prognostic,
                        local_state_auxiliary,
                        t,
                        (HorizontalDirection(),),
                    )

                    # Precomputing J ∇ξⁱ⋅ F
                    @unroll for s in 1:num_state_prognostic
                        F1, F2, F3 =
                            local_flux[1, s], local_flux[2, s], local_flux[3, s]
                        shared_flux[1, i, j, s] +=
                            M * (ξ1x1 * F1 + ξ1x2 * F2 + ξ1x3 * F3)
                        if dim == 3
                            shared_flux[2, i, j, s] +=
                                M * (ξ2x1 * F1 + ξ2x2 * F2 + ξ2x3 * F3)
                        end
                    end
                end
            end

            if dim == 3 && direction isa EveryDirection
                @unroll for n in 1:Nq3
                    MI = local_MI[n]
                    @unroll for s in 1:num_state_prognostic
                        local_tendency[n, s] += MI * D[k, n] * local_flux_3[s]
                    end
                end
            end

            # Computes the contribution due to the source term S
            if add_source
                fill!(local_source, -zero(eltype(local_source)))
                source_arr!(
                    balance_law,
                    local_source,
                    local_state_prognostic,
                    local_state_gradient_flux,
                    local_state_auxiliary,
                    t,
                    (model_direction,),
                )

                @unroll for s in 1:num_state_prognostic
                    local_tendency[k, s] += local_source[s]
                end

            end
            @synchronize

            # Weak "inside metrics" derivative.
            # Computes the rest of the volume term: M⁻¹DᵀF
            MI = local_MI[k]
            @unroll for s in 1:num_state_prognostic
                @unroll for n in 1:Nq1
                    # ξ1-grid lines
                    local_tendency[k, s] +=
                        MI * D[n, i] * shared_flux[1, n, j, s]

                    # ξ2-grid lines
                    if dim == 3 || (dim == 2 && direction isa EveryDirection)
                        local_tendency[k, s] +=
                            MI * D[n, j] * shared_flux[2, i, n, s]
                    end
                end
            end
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:num_state_prognostic
                if β != 0
                    T = α * local_tendency[k, s] + β * tendency[ijk, s, e]
                else
                    T = α * local_tendency[k, s]
                end
                tendency[ijk, s, e] = T
            end
        end
    end
end


@kernel function volume_tendency!(
    balance_law::BalanceLaw,
    ::Val{info},
    model_direction,
    ::VerticalDirection,
    tendency,
    state_prognostic,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    ω,
    D,
    elems,
    α,
    β,
    add_source = false,
) where {info}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_auxiliary = number_states(balance_law, Auxiliary())

        ngradlapstate = number_states(balance_law, GradientLaplacian())
        nhyperviscstate = number_states(balance_law, Hyperdiffusive())

        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        local_source = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_flux = MArray{Tuple{3, num_state_prognostic}, FT}(undef)
        local_flux_total = MArray{Tuple{3, num_state_prognostic}, FT}(undef)

        _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
        _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
        _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

        @inbounds Nqv = dim == 2 ? Nq2 : info.Nq[dim]
        shared_flux_size =
            dim == 2 ? (Nq1, Nqv, num_state_prognostic) : (0, 0, 0)
    end

    # Arrays for F, and the differentiation matrix D
    shared_flux = @localmem FT shared_flux_size

    # Storage for tendency and mass inverse M⁻¹
    local_tendency = @private FT (Nq3, num_state_prognostic)
    local_MI = @private FT (Nq3,)

    # Grab the index associated with the current element `e` and the
    # horizontal quadrature indices `i` (in the ξ1-direction),
    # `j` (in the ξ2-direction) [directions on the reference element].
    # Parallelize over elements, then over columns
    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            # initialize local tendency
            @unroll for s in 1:num_state_prognostic
                local_tendency[k, s] = zero(FT)
            end
            # read in mass matrix inverse for element `e`
            local_MI[k] = vgeo[ijk, _MI, e]
        end

        # ensure D is loaded
        @synchronize(dim == 3)

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            M = vgeo[ijk, _M, e]

            # Extract vertical Jacobian terms ∂ζ/∂xⱼ
            ζx1 = vgeo[ijk, _ζx1, e]
            ζx2 = vgeo[ijk, _ζx2, e]
            ζx3 = vgeo[ijk, _ζx3, e]

            # Read fields into registers (hopefully)
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end

            @unroll for s in 1:num_state_gradient_flux
                local_state_gradient_flux[s] = state_gradient_flux[ijk, s, e]
            end

            @unroll for s in 1:nhyperviscstate
                local_state_hyperdiffusion[s] = Qhypervisc_grad[ijk, s, e]
            end

            # Computes the local inviscid fluxes Fⁱⁿᵛ
            fill!(local_flux, -zero(eltype(local_flux)))
            flux_first_order_arr!(
                balance_law,
                local_flux,
                local_state_prognostic,
                local_state_auxiliary,
                t,
                (model_direction,),
            )

            @unroll for s in 1:num_state_prognostic
                local_flux_total[1, s] = local_flux[1, s]
                local_flux_total[2, s] = local_flux[2, s]
                local_flux_total[3, s] = local_flux[3, s]
            end

            # Computes the local viscous fluxes Fᵛⁱˢᶜ
            fill!(local_flux, -zero(eltype(local_flux)))
            flux_second_order_arr!(
                balance_law,
                local_flux,
                local_state_prognostic,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                local_state_auxiliary,
                t,
            )

            @unroll for s in 1:num_state_prognostic
                local_flux_total[1, s] += local_flux[1, s]
                local_flux_total[2, s] += local_flux[2, s]
                local_flux_total[3, s] += local_flux[3, s]
            end

            # Build "inside metrics" flux
            @unroll for s in 1:num_state_prognostic
                F1, F2, F3 = local_flux_total[1, s],
                local_flux_total[2, s],
                local_flux_total[3, s]
                Fv = M * (ζx1 * F1 + ζx2 * F2 + ζx3 * F3)
                if dim == 2
                    shared_flux[i, j, s] = Fv
                else
                    local_flux_total[1, s] = Fv
                end
            end

            # In the case of the remainder model we may need to loop through the
            # models to add in restricted direction components
            if model_direction isa EveryDirection && balance_law isa RemBL
                if rembl_has_subs_direction(VerticalDirection(), balance_law)
                    fill!(local_flux, -zero(eltype(local_flux)))
                    flux_first_order_arr!(
                        balance_law,
                        local_flux,
                        local_state_prognostic,
                        local_state_auxiliary,
                        t,
                        (VerticalDirection(),),
                    )

                    # Precomputing J ∇ζ⋅ F
                    @unroll for s in 1:num_state_prognostic
                        F1, F2, F3 =
                            local_flux[1, s], local_flux[2, s], local_flux[3, s]
                        Fv = M * (ζx1 * F1 + ζx2 * F2 + ζx3 * F3)
                        if dim == 2
                            shared_flux[i, j, s] += Fv
                        else
                            local_flux_total[1, s] += Fv
                        end
                    end
                end
            end

            if dim == 3
                @unroll for n in 1:Nq3
                    MI = local_MI[n]
                    @unroll for s in 1:num_state_prognostic
                        local_tendency[n, s] +=
                            MI * D[k, n] * local_flux_total[1, s]
                    end
                end
            end

            # Computes the contribution due to the source term S
            if add_source
                fill!(local_source, -zero(eltype(local_source)))
                source_arr!(
                    balance_law,
                    local_source,
                    local_state_prognostic,
                    local_state_gradient_flux,
                    local_state_auxiliary,
                    t,
                    (model_direction,),
                )

                @unroll for s in 1:num_state_prognostic
                    local_tendency[k, s] += local_source[s]
                end
            end
            @synchronize(dim == 2)

            # Weak "inside metrics" derivative.
            # Computes the rest of the volume term: M⁻¹DᵀMF
            if dim == 2
                MI = local_MI[k]
                @unroll for n in 1:Nqv
                    @unroll for s in 1:num_state_prognostic
                        local_tendency[k, s] +=
                            MI * D[n, j] * shared_flux[i, n, s]
                    end
                end
            end
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:num_state_prognostic
                if β != 0
                    T = α * local_tendency[k, s] + β * tendency[ijk, s, e]
                else
                    T = α * local_tendency[k, s]
                end
                tendency[ijk, s, e] = T
            end
        end
    end
end

@doc """
    function dgsem_interface_tendency!(
        balance_law::BalanceLaw,
        ::Val{dim},
        ::Val{polyorder},
        direction,
        numerical_flux_first_order,
        numerical_flux_second_order,
        tendency,
        state_prognostic,
        state_gradient_flux,
        Qhypervisc_grad,
        state_auxiliary,
        vgeo,
        sgeo,
        t,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        elems,
        α,
    )

Compute kernel for evaluating the interface tendencies for the
DG form:

∫ₑ ψ⋅ ∂q/∂t dx - ∫ₑ ∇ψ⋅(Fⁱⁿᵛ + Fᵛⁱˢᶜ) dx + ∮ₑ n̂ ψ⋅(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆) dS,

or equivalently in matrix form:

dQ/dt = M⁻¹(MS + DᵀM(Fⁱⁿᵛ + Fᵛⁱˢᶜ) + ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)).

This kernel computes the surface terms: M⁻¹ ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)),
where M is the mass matrix, Mf is the face mass matrix, L is an interpolator
from volume to face, and Fⁱⁿᵛ⋆, Fᵛⁱˢᶜ⋆
are the numerical fluxes for the inviscid and viscous
fluxes, respectively.
""" dgsem_interface_tendency!
@kernel function dgsem_interface_tendency!(
    balance_law::BalanceLaw,
    ::Val{info},
    direction,
    numerical_flux_first_order,
    numerical_flux_second_order,
    tendency,
    state_prognostic,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    α,
) where {info}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        nhyperviscstate = number_states(balance_law, Hyperdiffusive())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        nface = info.nface
        Np = info.Np
        Nqk = info.Nqk

        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        local_state_prognostic⁻ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux⁻ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion⁻ = MArray{Tuple{nhyperviscstate}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        # Need two copies since numerical_flux_first_order! can modify state_prognostic⁺
        local_state_prognostic⁺nondiff =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁺diff =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        # Need two copies since numerical_flux_first_order! can modify state_auxiliary⁺
        local_state_auxiliary⁺nondiff =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary⁺diff =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_gradient_flux⁺ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_hyperdiffusion⁺ = MArray{Tuple{nhyperviscstate}, FT}(undef)

        local_state_prognostic_bottom1 =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux_bottom1 =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_flux = MArray{Tuple{num_state_prognostic}, FT}(undef)
    end

    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    @inbounds for f in faces
        # The remainder model needs to know which direction of face the model is
        # being evaluated for. So faces 1:(nface - 2) are flagged as
        # `HorizontalDirection()` faces and the remaining two faces are
        # `VerticalDirection()` faces
        face_direction =
            f in 1:(nface - 2) ? (EveryDirection(), HorizontalDirection()) :
            (EveryDirection(), VerticalDirection())
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        bctag = elemtobndy[f, e⁻]
        # Get surface mass, volume mass inverse
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        if bctag != 0
            # TODO: we will use vmap⁺ to store the boundary element info
            #be = e⁺
            #bcid = vid⁺
            e⁺ = e⁻
            vid⁺ = vid⁻
        end

        # Load minus side data
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic⁻[s] = state_prognostic[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux⁻[s] = state_gradient_flux[vid⁻, s, e⁻]
        end

        @unroll for s in 1:nhyperviscstate
            local_state_hyperdiffusion⁻[s] = Qhypervisc_grad[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁻[s] = state_auxiliary[vid⁻, s, e⁻]
        end

        # Load plus side data
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic⁺diff[s] =
                local_state_prognostic⁺nondiff[s] =
                    state_prognostic[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux⁺[s] = state_gradient_flux[vid⁺, s, e⁺]
        end

        @unroll for s in 1:nhyperviscstate
            local_state_hyperdiffusion⁺[s] = Qhypervisc_grad[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁺diff[s] =
                local_state_auxiliary⁺nondiff[s] = state_auxiliary[vid⁺, s, e⁺]
        end

        # Oh dang, it's boundary conditions
        fill!(local_flux, -zero(eltype(local_flux)))
        if bctag == 0
            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                SVector(normal_vector),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁻,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁺nondiff,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁺nondiff,
                ),
                t,
                face_direction,
            )
            numerical_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                normal_vector,
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁻,
                ),
                Vars{vars_state(balance_law, GradientFlux(), FT)}(
                    local_state_gradient_flux⁻,
                ),
                Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                    local_state_hyperdiffusion⁻,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁺diff,
                ),
                Vars{vars_state(balance_law, GradientFlux(), FT)}(
                    local_state_gradient_flux⁺,
                ),
                Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                    local_state_hyperdiffusion⁺,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁺diff,
                ),
                t,
            )
        else
            if (dim == 2 && f == 3) || (dim == 3 && f == 5)
                if info.N[end] == 0
                    # Loop up to next element for all horizontal elements
                    @unroll for s in 1:num_state_prognostic
                        local_state_prognostic_bottom1[s] =
                            state_prognostic[n, s, e⁻ + 1]
                    end
                    @unroll for s in 1:num_state_gradient_flux
                        local_state_gradient_flux_bottom1[s] =
                            state_gradient_flux[n, s, e⁻ + 1]
                    end
                    @unroll for s in 1:num_state_auxiliary
                        local_state_auxiliary_bottom1[s] =
                            state_auxiliary[n, s, e⁻ + 1]
                    end
                else
                    # Loop up the first element along all horizontal elements
                    @unroll for s in 1:num_state_prognostic
                        local_state_prognostic_bottom1[s] =
                            state_prognostic[n + Nqk^2, s, e⁻]
                    end
                    @unroll for s in 1:num_state_gradient_flux
                        local_state_gradient_flux_bottom1[s] =
                            state_gradient_flux[n + Nqk^2, s, e⁻]
                    end
                    @unroll for s in 1:num_state_auxiliary
                        local_state_auxiliary_bottom1[s] =
                            state_auxiliary[n + Nqk^2, s, e⁻]
                    end
                end
            end

            bcs = boundary_conditions(balance_law)
            # TODO: there is probably a better way to unroll this loop
            Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
                bc = bcs[d]
                numerical_boundary_flux_first_order!(
                    numerical_flux_first_order,
                    bc,
                    balance_law,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                    SVector(normal_vector),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁻,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁻,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁺nondiff,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁺nondiff,
                    ),
                    t,
                    face_direction,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic_bottom1,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary_bottom1,
                    ),
                )
                numerical_boundary_flux_second_order!(
                    numerical_flux_second_order,
                    bc,
                    balance_law,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(local_flux),
                    normal_vector,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁻,
                    ),
                    Vars{vars_state(balance_law, GradientFlux(), FT)}(
                        local_state_gradient_flux⁻,
                    ),
                    Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                        local_state_hyperdiffusion⁻,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁻,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁺diff,
                    ),
                    Vars{vars_state(balance_law, GradientFlux(), FT)}(
                        local_state_gradient_flux⁺,
                    ),
                    Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                        local_state_hyperdiffusion⁺,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁺diff,
                    ),
                    t,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic_bottom1,
                    ),
                    Vars{vars_state(balance_law, GradientFlux(), FT)}(
                        local_state_gradient_flux_bottom1,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary_bottom1,
                    ),
                )
            end d -> throw(BoundsError(bcs, bctag))
        end

        # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆))
        @unroll for s in 1:num_state_prognostic
            # FIXME: Should we pretch these?
            tendency[vid⁻, s, e⁻] -= α * vMI * sM * local_flux[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

"""
    function volume_gradients!(
        balance_law::BalanceLaw,
        ::Val{dim},
        ::Val{polyorder},
        direction,
        state_prognostic,
        state_gradient_flux,
        Qhypervisc_grad,
        state_auxiliary,
        vgeo,
        t,
        D,
        ::Val{hypervisc_indexmap},
        elems,
        increment = false,
    )

Computes the volume integral for the auxiliary equation
(in DG strong form):

∫ₑ ψI⋅Σ dx = ∫ₑ ψI⋅∇G dx + ∮ₑ nψI⋅(G* - G) dS,

or equivalently in matrix notation:

Σ = M⁻¹ LᵀMf(G* - G) + D G

This kernel computes the volume gradient: D * G, where
D is the differentiation matrix and G is the auxiliary
gradient flux.
"""
@kernel function volume_gradients!(
    balance_law::BalanceLaw,
    ::Val{info},
    direction,
    state_prognostic,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    D,
    ::Val{hypervisc_indexmap},
    elems,
    increment = false,
) where {info, hypervisc_indexmap}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        ngradstate = number_states(balance_law, Gradient())
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_auxiliary = number_states(balance_law, Auxiliary())

        # Kernel assumes same polynomial order in both
        # horizontal directions (x, y)
        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        ngradtransformstate = num_state_prognostic

        local_transform = MArray{Tuple{ngradstate}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
    end

    # Transformation from conservative variables to
    # primitive variables (i.e. ρu → u)
    shared_transform = @localmem FT (Nq1, Nq2, ngradstate)

    local_state_prognostic = @private FT (ngradtransformstate, Nq3)
    local_state_auxiliary = @private FT (num_state_auxiliary, Nq3)
    local_transform_gradient = @private FT (3, ngradstate, Nq3)
    Gξ3 = @private FT (ngradstate, Nq3)

    # Grab the index associated with the current element `e` and the
    # horizontal quadrature indices `i` (in the ξ1-direction),
    # `j` (in the ξ2-direction) [directions on the reference element].
    # Parallelize over elements, then over columns
    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds @views begin
        @unroll for k in 1:Nq3
            # Initialize local gradient variables
            @unroll for s in 1:ngradstate
                local_transform_gradient[1, s, k] = -zero(FT)
                local_transform_gradient[2, s, k] = -zero(FT)
                local_transform_gradient[3, s, k] = -zero(FT)
                Gξ3[s, k] = -zero(FT)
            end

            # Load prognostic and auxiliary variables
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradtransformstate
                local_state_prognostic[s, k] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s, k] = state_auxiliary[ijk, s, e]
            end
        end

        # Compute G(q) and write the result into shared memory
        @unroll for k in 1:Nq3
            fill!(local_transform, -zero(eltype(local_transform)))
            compute_gradient_argument_arr!(
                balance_law,
                local_transform,
                local_state_prognostic[:, k],
                local_state_auxiliary[:, k],
                t,
            )

            @unroll for s in 1:ngradstate
                shared_transform[i, j, s] = local_transform[s]
            end

            # Synchronize threads on the device
            @synchronize

            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            ξ1x1, ξ1x2, ξ1x3 =
                vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]

            # Compute gradient of each state
            @unroll for s in 1:ngradstate
                Gξ1 = Gξ2 = zero(FT)

                @unroll for n in 1:Nq1
                    # Smack G with the differentiation matrix
                    Gξ1 += D[i, n] * shared_transform[n, j, s]
                    if dim == 3 || (dim == 2 && direction isa EveryDirection)
                        Gξ2 += D[j, n] * shared_transform[i, n, s]
                    end
                    # Compute the gradient of G over the entire column
                    if dim == 3 && direction isa EveryDirection
                        Gξ3[s, n] += D[n, k] * shared_transform[i, j, s]
                    end
                end

                # Application of chain-rule in ξ1 and ξ2 directions,
                # ∂G/∂xi = ∂ξ1/∂xi * ∂G/∂ξ1, ∂G/∂xi = ∂ξ2/∂xi * ∂G/∂ξ2
                # to get a physical gradient
                local_transform_gradient[1, s, k] += ξ1x1 * Gξ1
                local_transform_gradient[2, s, k] += ξ1x2 * Gξ1
                local_transform_gradient[3, s, k] += ξ1x3 * Gξ1

                if dim == 3 || (dim == 2 && direction isa EveryDirection)
                    ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e],
                    vgeo[ijk, _ξ2x2, e],
                    vgeo[ijk, _ξ2x3, e]
                    local_transform_gradient[1, s, k] += ξ2x1 * Gξ2
                    local_transform_gradient[2, s, k] += ξ2x2 * Gξ2
                    local_transform_gradient[3, s, k] += ξ2x3 * Gξ2
                end
            end

            # Synchronize threads on the device
            @synchronize
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            # Application of chain-rule in ξ3-direction: ∂G/∂xi = ∂ξ3/∂xi * ∂G/∂ξ3
            if dim == 3 && direction isa EveryDirection
                ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e],
                vgeo[ijk, _ξ3x2, e],
                vgeo[ijk, _ξ3x3, e]
                @unroll for s in 1:ngradstate
                    local_transform_gradient[1, s, k] += ξ3x1 * Gξ3[s, k]
                    local_transform_gradient[2, s, k] += ξ3x2 * Gξ3[s, k]
                    local_transform_gradient[3, s, k] += ξ3x3 * Gξ3[s, k]
                end
            end

            # Hyperdiffusion (avoid recomputing gradients of the state since
            # these are needed for the hyperdiffusion kernels)
            @unroll for s in 1:ngradlapstate
                if increment
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e] +=
                        local_transform_gradient[1, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e] +=
                        local_transform_gradient[2, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e] +=
                        local_transform_gradient[3, hypervisc_indexmap[s], k]
                else
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e] =
                        local_transform_gradient[1, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e] =
                        local_transform_gradient[2, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e] =
                        local_transform_gradient[3, hypervisc_indexmap[s], k]
                end
            end

            if num_state_gradient_flux > 0
                fill!(
                    local_state_gradient_flux,
                    -zero(eltype(local_state_gradient_flux)),
                )

                # Applies a linear transformation of gradients to the diffusive variables
                compute_gradient_flux_arr!(
                    balance_law,
                    local_state_gradient_flux,
                    local_transform_gradient[:, :, k],
                    local_state_prognostic[:, k],
                    local_state_auxiliary[:, k],
                    t,
                )

                # Write out the result of the kernel to global memory
                @unroll for s in 1:num_state_gradient_flux
                    if increment
                        state_gradient_flux[ijk, s, e] +=
                            local_state_gradient_flux[s]
                    else
                        state_gradient_flux[ijk, s, e] =
                            local_state_gradient_flux[s]
                    end
                end
            end
        end
    end
end

@kernel function volume_gradients!(
    balance_law::BalanceLaw,
    ::Val{info},
    ::VerticalDirection,
    state_prognostic,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    t,
    D,
    ::Val{hypervisc_indexmap},
    elems,
    increment = false,
) where {info, hypervisc_indexmap}
    @uniform begin
        dim = info.dim

        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        ngradstate = number_states(balance_law, Gradient())
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_auxiliary = number_states(balance_law, Auxiliary())

        # Assumes same polynomial order in both
        # horizontal directions (x,y)
        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        ngradtransformstate = num_state_prognostic

        local_transform = MArray{Tuple{ngradstate}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)

        _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
        _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
        _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

        Gζ_size = dim == 3 ? (ngradstate, Nq3) : (0, 0)
        @inbounds Nqv = dim == 2 ? Nq2 : info.Nq[dim]
        shared_transform_dim2 = dim == 2 ? Nqv : Nq1
    end

    # Transformation from conservative variables to
    # primitive variables (i.e. ρu → u)
    shared_transform = @localmem FT (Nq1, shared_transform_dim2, ngradstate)

    local_state_prognostic = @private FT (ngradtransformstate, Nq3)
    local_state_auxiliary = @private FT (num_state_auxiliary, Nq3)
    local_transform_gradient = @private FT (3, ngradstate, Nq3)

    local_ζ = @private FT (3, Nq3)

    Gζ = @private FT Gζ_size

    # Grab the index associated with the current element `e` and the
    # horizontal quadrature indices `i` (in the ξ1-direction),
    # `j` (in the ξ2-direction) [directions on the reference element].
    # Parallelize over elements, then over columns
    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds @views begin

        @unroll for k in 1:Nq3
            # Initialize local gradient variables
            @unroll for s in 1:ngradstate
                local_transform_gradient[1, s, k] = -zero(FT)
                local_transform_gradient[2, s, k] = -zero(FT)
                local_transform_gradient[3, s, k] = -zero(FT)
                if dim == 3
                    Gζ[s, k] = -zero(FT)
                end
            end

            # Load prognostic and auxiliary variables
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradtransformstate
                local_state_prognostic[s, k] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s, k] = state_auxiliary[ijk, s, e]
            end

            # Load geometry terms for the Jacobian: ∂ζ/∂xⱼ
            local_ζ[1, k] = vgeo[ijk, _ζx1, e]
            local_ζ[2, k] = vgeo[ijk, _ζx2, e]
            local_ζ[3, k] = vgeo[ijk, _ζx3, e]
        end

        # Compute G(q) and write the result into shared memory
        @unroll for k in 1:Nq3
            fill!(local_transform, -zero(eltype(local_transform)))
            compute_gradient_argument_arr!(
                balance_law,
                local_transform,
                local_state_prognostic[:, k],
                local_state_auxiliary[:, k],
                t,
            )

            @unroll for s in 1:ngradstate
                shared_transform[i, j, s] = local_transform[s]
            end

            # Synchronize threads on the device
            @synchronize

            # Compute gradient of each state
            @unroll for s in 1:ngradstate
                # Compute the gradient of G using the chain-rule:
                # ∂G/∂xi = ∂ζ/∂xi * ∂G/∂ζ to get a physical gradient
                if dim == 2
                    Gζ1 = zero(FT)
                    @unroll for n in 1:Nqv
                        Gζ1 += D[j, n] * shared_transform[i, n, s]
                    end
                    local_transform_gradient[1, s, k] += local_ζ[1, k] * Gζ1
                    local_transform_gradient[2, s, k] += local_ζ[2, k] * Gζ1
                    local_transform_gradient[3, s, k] += local_ζ[3, k] * Gζ1
                else
                    @unroll for n in 1:Nq3
                        Gζ[s, n] += D[n, k] * shared_transform[i, j, s]
                    end
                end
            end
            @synchronize
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            # Application of chain-rule: ∂G/∂xi = ∂ζ/∂xi * ∂G/∂ζ
            if dim == 3
                @unroll for s in 1:ngradstate
                    local_transform_gradient[1, s, k] +=
                        local_ζ[1, k] * Gζ[s, k]
                    local_transform_gradient[2, s, k] +=
                        local_ζ[2, k] * Gζ[s, k]
                    local_transform_gradient[3, s, k] +=
                        local_ζ[3, k] * Gζ[s, k]
                end
            end

            # Hyperdiffusion (avoid recomputing gradients of the state since
            # these are needed for the hyperdiffusion kernels)
            @unroll for s in 1:ngradlapstate
                if increment
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e] +=
                        local_transform_gradient[1, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e] +=
                        local_transform_gradient[2, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e] +=
                        local_transform_gradient[3, hypervisc_indexmap[s], k]
                else
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e] =
                        local_transform_gradient[1, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e] =
                        local_transform_gradient[2, hypervisc_indexmap[s], k]
                    Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e] =
                        local_transform_gradient[3, hypervisc_indexmap[s], k]
                end
            end

            if num_state_gradient_flux > 0
                fill!(
                    local_state_gradient_flux,
                    -zero(eltype(local_state_gradient_flux)),
                )

                # Applies a linear transformation of gradients to the diffusive variables
                compute_gradient_flux_arr!(
                    balance_law,
                    local_state_gradient_flux,
                    local_transform_gradient[:, :, k],
                    local_state_prognostic[:, k],
                    local_state_auxiliary[:, k],
                    t,
                )

                # Write out the result of the kernel to global memory
                @unroll for s in 1:num_state_gradient_flux
                    if increment
                        state_gradient_flux[ijk, s, e] +=
                            local_state_gradient_flux[s]
                    else
                        state_gradient_flux[ijk, s, e] =
                            local_state_gradient_flux[s]
                    end
                end
            end
        end
    end
end

@doc """
    function dgsem_interface_gradients!(
        balance_law::BalanceLaw,
        ::Val{dim},
        ::Val{polyorder},
        direction,
        numerical_flux_gradient,
        state_prognostic,
        state_gradient_flux,
        Qhypervisc_grad,
        state_auxiliary,
        vgeo,
        sgeo,
        t,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        ::Val{hypervisc_indexmap},
        elems,
    )

Computes the surface integral for the auxiliary equation
(in DG strong form):

∫ₑ ψI⋅Σ dx = ∫ₑ ψI⋅∇G dx + ∮ₑ nψI⋅(G* - G) dS,

or equivalently in matrix notation:

Σ = M⁻¹ LᵀMf(G* - G) + D G

This kernel computes the interface gradient term: M⁻¹ LᵀMf(G* - G),
where M is the mass matrix, Mf is the face mass matrix, L is an interpolator
from volume to face, G is the
auxiliary gradient flux, and G* is the associated numerical flux.
""" dgsem_interface_gradients!
@kernel function dgsem_interface_gradients!(
    balance_law::BalanceLaw,
    ::Val{info},
    direction,
    numerical_flux_gradient,
    state_prognostic,
    state_gradient_flux,
    Qhypervisc_grad,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    ::Val{hypervisc_indexmap},
    elems,
) where {info, hypervisc_indexmap}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        ngradstate = number_states(balance_law, Gradient())
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        nface = info.nface
        Np = info.Np
        Nqk = info.Nqk

        # Determines the number of faces depending on
        # the direction argument
        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        ngradtransformstate = num_state_prognostic

        # Create local arrays for states inside the element, wrt to a particular face (-)
        local_state_prognostic⁻ = MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_transform⁻ = MArray{Tuple{ngradstate}, FT}(undef)
        l_nG⁻ = MArray{Tuple{3, ngradstate}, FT}(undef)

        # Create local arrays for states outside the element, wrt to a particular face (+)
        local_state_prognostic⁺ = MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁺ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_transform⁺ = MArray{Tuple{ngradstate}, FT}(undef)

        # FIXME state_gradient_flux is sort of a terrible name...
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_transform_gradient = MArray{Tuple{3, ngradstate}, FT}(undef)
        local_state_prognostic⁻visc =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)

        local_state_prognostic_bottom1 =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
    end

    # Element index
    eI = @index(Group, Linear)
    # Index of a quadrature point on a face
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    # Spin over the faces of the element `e`
    @inbounds for f in faces
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )

        # Extract surface mass operator `sM` and volumne mass inverse `vMI`
        bctag = elemtobndy[f, e⁻]
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        if bctag != 0
            # TODO: we will use vmap⁺ to store the boundary element info
            #be = e⁺
            #bcid = vid⁺
            e⁺ = e⁻
            vid⁺ = vid⁻
        end

        # Load minus side data
        @unroll for s in 1:ngradtransformstate
            local_state_prognostic⁻[s] = state_prognostic[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁻[s] = state_auxiliary[vid⁻, s, e⁻]
        end

        # Compute G(q) on the minus side write the result into registers
        fill!(local_transform⁻, -zero(eltype(local_transform⁻)))
        compute_gradient_argument_arr!(
            balance_law,
            local_transform⁻,
            local_state_prognostic⁻,
            local_state_auxiliary⁻,
            t,
        )

        # Load plus side data
        @unroll for s in 1:ngradtransformstate
            local_state_prognostic⁺[s] = state_prognostic[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁺[s] = state_auxiliary[vid⁺, s, e⁺]
        end

        # Compute G(q) on the plus side and write the result into registers
        fill!(local_transform⁺, -zero(eltype(local_transform⁺)))
        compute_gradient_argument_arr!(
            balance_law,
            local_transform⁺,
            local_state_prognostic⁺,
            local_state_auxiliary⁺,
            t,
        )

        # Oh drat, it's boundary conditions

        fill!(
            local_state_gradient_flux,
            -zero(eltype(local_state_gradient_flux)),
        )
        if bctag == 0  # Periodic boundary condition (boundary-less)
            # Computes G* on the minus side
            numerical_flux_gradient!(
                numerical_flux_gradient,
                balance_law,
                local_transform_gradient,
                normal_vector,
                local_transform⁻,
                local_state_prognostic⁻,
                local_state_auxiliary⁻,
                local_transform⁺,
                local_state_prognostic⁺,
                local_state_auxiliary⁺,
                t,
            )
            if num_state_gradient_flux > 0
                # Applies linear transformation of gradients to the diffusive variables
                # on the minus side
                compute_gradient_flux_arr!(
                    balance_law,
                    local_state_gradient_flux,
                    local_transform_gradient,
                    local_state_prognostic⁻,
                    local_state_auxiliary⁻,
                    t,
                )
            end
        else
            # NOTE: Used for boundary conditions related to the energy
            # variables (see `BulkFormulaEnergy`)
            if (dim == 2 && f == 3) || (dim == 3 && f == 5)
                if info.N[end] == 0
                    # Loop up to next element for all horizontal elements
                    @unroll for s in 1:num_state_prognostic
                        local_state_prognostic_bottom1[s] =
                            state_prognostic[n, s, e⁻ + 1]
                    end
                    @unroll for s in 1:num_state_auxiliary
                        local_state_auxiliary_bottom1[s] =
                            state_auxiliary[n, s, e⁻ + 1]
                    end
                else
                    # Loop up the first element along all horizontal elements
                    @unroll for s in 1:num_state_prognostic
                        local_state_prognostic_bottom1[s] =
                            state_prognostic[n + Nqk^2, s, e⁻]
                    end
                    @unroll for s in 1:num_state_auxiliary
                        local_state_auxiliary_bottom1[s] =
                            state_auxiliary[n + Nqk^2, s, e⁻]
                    end
                end
            end
            bcs = boundary_conditions(balance_law)
            # TODO: there is probably a better way to unroll this loop
            Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
                bc = bcs[d]
                # Computes G* incorporating boundary conditions
                numerical_boundary_flux_gradient!(
                    numerical_flux_gradient,
                    bc,
                    balance_law,
                    local_transform_gradient,
                    SVector(normal_vector),
                    Vars{vars_state(balance_law, Gradient(), FT)}(
                        local_transform⁻,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁻,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁻,
                    ),
                    Vars{vars_state(balance_law, Gradient(), FT)}(
                        local_transform⁺,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁺,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁺,
                    ),
                    t,
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic_bottom1,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary_bottom1,
                    ),
                )
                if num_state_gradient_flux > 0
                    # Applies linear transformation of gradients to the diffusive variables
                    # on the minus side
                    compute_gradient_flux_arr!(
                        balance_law,
                        local_state_gradient_flux,
                        local_transform_gradient,
                        local_state_prognostic⁻,
                        local_state_auxiliary⁻,
                        t,
                    )
                end
            end d -> throw(BoundsError(bcs, bctag))
        end

        # Compute n*G, where n is the face normal
        @unroll for j in 1:ngradstate
            @unroll for i in 1:3
                l_nG⁻[i, j] = normal_vector[i] * local_transform⁻[j]
            end
        end

        # Storing gradients for applying hyperdiffusion
        @unroll for s in 1:ngradlapstate
            j = hypervisc_indexmap[s]
            Qhypervisc_grad[vid⁻, 3 * (s - 1) + 1, e⁻] +=
                vMI * sM * (local_transform_gradient[1, j] - l_nG⁻[1, j])
            Qhypervisc_grad[vid⁻, 3 * (s - 1) + 2, e⁻] +=
                vMI * sM * (local_transform_gradient[2, j] - l_nG⁻[2, j])
            Qhypervisc_grad[vid⁻, 3 * (s - 1) + 3, e⁻] +=
                vMI * sM * (local_transform_gradient[3, j] - l_nG⁻[3, j])
        end

        # Applies linear transformation of gradients to the diffusive variables
        # on the minus side
        compute_gradient_flux_arr!(
            balance_law,
            local_state_prognostic⁻visc,
            l_nG⁻,
            local_state_prognostic⁻,
            local_state_auxiliary⁻,
            t,
        )

        # This is the surface integral evaluated discretely
        # M^(-1) Mf(G* - G)
        @unroll for s in 1:num_state_gradient_flux
            state_gradient_flux[vid⁻, s, e⁻] +=
                vMI *
                sM *
                (local_state_gradient_flux[s] - local_state_prognostic⁻visc[s])
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function kernel_init_state_prognostic!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    state,
    state_auxiliary,
    vgeo,
    elems,
    args...,
) where {dim, polyorder}
    N = polyorder
    FT = eltype(state_auxiliary)
    num_state_auxiliary = number_states(balance_law, Auxiliary())
    num_state_prognostic = number_states(balance_law, Prognostic())

    Nq = N .+ 1
    @inbounds Nqk = dim == 2 ? 1 : Nq[dim]
    @inbounds Np = Nq[1] * Nq[2] * Nqk

    l_state = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end
        @unroll for s in 1:num_state_prognostic
            l_state[s] = state[n, s, e]
        end
        init_state_prognostic_arr!(
            balance_law,
            l_state,
            local_state_auxiliary,
            LocalGeometry{Np, N}(vgeo, n, e),
            args...,
        )
        @unroll for s in 1:num_state_prognostic
            state[n, s, e] = l_state[s]
        end
        @unroll for s in 1:num_state_auxiliary
            state_auxiliary[n, s, e] = local_state_auxiliary[s]
        end
    end
end


@doc """
    kernel_nodal_init_state_auxiliary!(balance_law::BalanceLaw, Val(polyorder),
                                       init_f!, state_auxiliary, state_init,
                                       Val(vars_state_init), vgeo, elems)

Computational kernel: Initialize the auxiliary state

See [`BalanceLaw`](@ref) for usage.
""" kernel_nodal_init_state_auxiliary!
@kernel function kernel_nodal_init_state_auxiliary!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{polyorder},
    init_f!,
    state_auxiliary,
    state_temporary,
    ::Val{vars_state_temporary},
    vgeo,
    elems,
) where {dim, polyorder, vars_state_temporary}
    N = polyorder
    FT = eltype(state_auxiliary)
    num_state_auxiliary = number_states(balance_law, Auxiliary())
    num_state_temporary = varsize(vars_state_temporary)

    Nq = N .+ 1
    @inbounds Nqk = dim == 2 ? 1 : Nq[dim]
    @inbounds Np = Nq[1] * Nq[2] * Nqk

    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    local_state_temporary = MArray{Tuple{num_state_temporary}, FT}(undef)

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_temporary
            local_state_temporary[s] = state_temporary[n, s, e]
        end

        init_f!(
            balance_law,
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                local_state_auxiliary,
            ),
            Vars{vars_state_temporary}(local_state_temporary),
            LocalGeometry{Np, N}(vgeo, n, e),
        )

        @unroll for s in 1:num_state_auxiliary
            state_auxiliary[n, s, e] = local_state_auxiliary[s]
        end
    end
end

@doc """
    kernel_nodal_update_auxiliary_state!(balance_law::BalanceLaw, ::Val{dim}, ::Val{N}, f!, state_prognostic, state_auxiliary, [state_gradient_flux,]
                          t, elems, activedofs) where {dim, N}

Update the auxiliary state array
""" kernel_nodal_update_auxiliary_state!
@kernel function kernel_nodal_update_auxiliary_state!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    f!,
    state_prognostic,
    state_auxiliary,
    t,
    elems,
    activedofs,
) where {dim, N}
    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(balance_law, Prognostic())
    num_state_auxiliary = number_states(balance_law, Auxiliary())

    Nq = N .+ 1
    @inbounds Nqk = dim == 2 ? 1 : Nq[dim]
    @inbounds Np = Nq[1] * Nq[2] * Nqk

    local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]

        active = activedofs[n + (e - 1) * Np]

        if active
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[n, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[n, s, e]
            end

            f!(
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary,
                ),
                t,
            )

            @unroll for s in 1:num_state_auxiliary
                state_auxiliary[n, s, e] = local_state_auxiliary[s]
            end
        end
    end
end

@kernel function kernel_nodal_update_auxiliary_state!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    f!,
    state_prognostic,
    state_auxiliary,
    state_gradient_flux,
    t,
    elems,
    activedofs,
) where {dim, N}
    FT = eltype(state_prognostic)
    num_state_prognostic = number_states(balance_law, Prognostic())
    num_state_gradient_flux = number_states(balance_law, GradientFlux())
    num_state_auxiliary = number_states(balance_law, Auxiliary())

    Nq = N .+ 1
    @inbounds Nqk = dim == 2 ? 1 : Nq[dim]
    @inbounds Np = Nq[1] * Nq[2] * Nqk

    local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    local_state_gradient_flux =
        MArray{Tuple{num_state_gradient_flux}, FT}(undef)

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]

        active = activedofs[n + (e - 1) * Np]

        if active
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[n, s, e]
            end

            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[n, s, e]
            end

            @unroll for s in 1:num_state_gradient_flux
                local_state_gradient_flux[s] = state_gradient_flux[n, s, e]
            end

            f!(
                balance_law,
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary,
                ),
                Vars{vars_state(balance_law, GradientFlux(), FT)}(
                    local_state_gradient_flux,
                ),
                t,
            )

            @unroll for s in 1:num_state_auxiliary
                state_auxiliary[n, s, e] = local_state_auxiliary[s]
            end
        end
    end
end

@doc """
    kernel_indefinite_stack_integral!(balance_law::BalanceLaw, ::Val{dim}, ::Val{N},
                                  ::Val{nvertelem}, state_prognostic, state_auxiliary, vgeo,
                                  Imat, elems) where {dim, N, nvertelem}
Computational kernel: compute indefinite integral along the vertical stack
See [`BalanceLaw`](@ref) for usage.
""" kernel_indefinite_stack_integral!
@kernel function kernel_indefinite_stack_integral!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    state_prognostic,
    state_auxiliary,
    vgeo,
    Imat,
    elems,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        nout = number_states(balance_law, UpwardIntegrals())

        # Number of Gauss-Lobatto quadrature points in each direction
        Nq = N .+ 1
        Nq1 = Nq[1]
        Nq2 = dim == 2 ? 1 : Nq[2]
        Nq3 = Nq[end]

        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_kernel = MArray{Tuple{nout, Nq3}, FT}(undef)
    end

    local_integral = @private FT (nout, Nq1)
    s_I = @localmem FT (Nq3, Nq3)

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for iv in i:Nq1:Nq3
            @unroll for jv in j:Nq2:Nq3
                s_I[iv, jv] = Imat[iv, jv]
            end
        end
        @synchronize

        # Initialize the constant state at zero
        @unroll for k in 1:Nq3
            @unroll for s in 1:nout
                local_integral[s, k] = 0
            end
        end

        eh = elems[_eh]

        # Loop up the stack of elements
        for ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem

            # Evaluate the integral kernel at each DOF in the slabk
            # loop up the pencil
            @unroll for k in 1:Nq3
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                Jc = vgeo[ijk, _JcV, e]
                @unroll for s in 1:num_state_prognostic
                    local_state_prognostic[s] = state_prognostic[ijk, s, e]
                end

                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end

                integral_load_auxiliary_state_arr!(
                    balance_law,
                    view(local_kernel, :, k),
                    local_state_prognostic,
                    local_state_auxiliary,
                )

                # Multiply in the curve jacobian
                @unroll for s in 1:nout
                    local_kernel[s, k] *= Jc
                end
            end

            # Evaluate the integral up the element
            @unroll for s in 1:nout
                @unroll for k in 1:Nq3
                    @unroll for n in 1:Nq3
                        local_integral[s, k] += s_I[k, n] * local_kernel[s, n]
                    end
                end
            end

            # Store out to memory and reset the background value for next element
            @unroll for k in 1:Nq3
                @unroll for s in 1:nout
                    local_kernel[s, k] = local_integral[s, k]
                end
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                integral_set_auxiliary_state_arr!(
                    balance_law,
                    view(state_auxiliary, ijk, :, e),
                    view(local_kernel, :, k),
                )
                @unroll for ind_out in 1:nout
                    local_integral[ind_out, k] = local_integral[ind_out, Nq3]
                end
            end
        end
    end
end

@kernel function kernel_reverse_indefinite_stack_integral!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    ::Val{nvertelem},
    state,
    state_auxiliary,
    elems,
) where {dim, N, nvertelem}
    @uniform begin
        FT = eltype(state_auxiliary)

        # Number of Gauss-Lobatto quadrature points in each direction
        Nq = N .+ 1
        Nq1 = Nq[1]
        Nq2 = dim == 2 ? 1 : Nq[2]
        Nq3 = Nq[end]
        nout = number_states(balance_law, DownwardIntegrals())

        # Note that k is the second not 4th index (since this is scratch memory and k
        # needs to be persistent across threads)
        l_T = MArray{Tuple{nout}, FT}(undef)
        l_V = MArray{Tuple{nout}, FT}(undef)
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]

        # Extract the top value from the top element which is degree of freedom
        # (i, j, Nq3)
        ijk = i + Nq1 * ((j - 1) + Nq2 * (Nq3 - 1))
        et = nvertelem + (eh - 1) * nvertelem
        reverse_integral_load_auxiliary_state_arr!(
            balance_law,
            l_T,
            view(state, ijk, :, et),
            view(state_auxiliary, ijk, :, et),
        )

        # Loop up the stack of elements
        #
        # In the case of N = 0 the forward integral computed the top face
        # integral value, when reversing we want to store the bottom face value
        # (there is no need to store the top face value of the top element since
        # the reverse integral will be zero there, whereas in the forward case
        # the bottom face of the first element was zero).
        #
        # This loop gets complicated in this case since we have a shifting of
        # the element values.

        # Loop limits for N = 0 versus N > 0
        for ev in (Nq3 == 1 ? (1:(nvertelem - 1)) : (1:nvertelem))
            e = ev + (eh - 1) * nvertelem
            @unroll for k in 1:Nq3
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                reverse_integral_load_auxiliary_state_arr!(
                    balance_law,
                    l_V,
                    view(state, ijk, :, e),
                    view(state_auxiliary, ijk, :, e),
                )
                l_V .= l_T .- l_V
                reverse_integral_set_auxiliary_state_arr!(
                    balance_law,
                    view(
                        state_auxiliary,
                        ijk,
                        :,
                        # In the N = 0 case we shift the data up
                        Nq3 == 1 ? e + 1 : e,
                    ),
                    l_V,
                )
            end
        end
        # We need to update the first vertical element value still with the very
        # top value
        if Nq3 == 1
            ev = 1
            k = 1
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            e = ev + (eh - 1) * nvertelem
            reverse_integral_set_auxiliary_state_arr!(
                balance_law,
                view(state_auxiliary, ijk, :, e),
                l_T,
            )
        end
    end
end

"""
    function volume_divergence_of_gradients!(
        balance_law::BalanceLaw,
        ::Val{info},
        direction,
        Qhypervisc_grad,
        Qhypervisc_div,
        vgeo,
        D,
        elems,
        increment = false,
    )

Compute kernel for evaluating the volume divergence of gradients (or,
equivalently, the scalar laplacian) for the DG form:

∫ₑ ψ⋅ΔG dx - ∫ₑ ∇ψ⋅∇G dx + ∮ₑ n̂ ψ⋅(∇G)⋆ dS,

or equivalently in matrix form:

ΔG = M⁻¹(DᵀM ∇G + ∑ᶠ LᵀMf (∇G)⋆).

This kernel computes the volume terms: M⁻¹(DᵀM ∇G),
where M is the mass matrix and D is the differentiation matrix,
and ∇G are the gradients.
"""
@kernel function volume_divergence_of_gradients!(
    balance_law::BalanceLaw,
    ::Val{info},
    direction,
    Qhypervisc_grad,
    Qhypervisc_div,
    vgeo,
    D,
    elems,
    increment = false,
) where {info}
    @uniform begin
        dim = info.dim
        FT = eltype(Qhypervisc_grad)
        ngradlapstate = number_states(balance_law, GradientLaplacian())

        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk
    end

    s_grad = @localmem FT (2, Nq1, Nq2, ngradlapstate)

    local_div = @private FT (Nq3, ngradlapstate)
    local_MI = @private FT (Nq3,)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            # initialize local tendency
            @unroll for s in 1:ngradlapstate
                local_div[k, s] = zero(FT)
            end
            # read in mass matrix inverse for element `e`
            local_MI[k] = vgeo[ijk, _MI, e]
        end

        @unroll for k in 1:Nq3
            @synchronize

            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            M = vgeo[ijk, _M, e]

            # Extract Jacobian terms ∂ξᵢ/∂xⱼ
            ξ1x1, ξ1x2, ξ1x3 =
                vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]
            if dim == 3 || (dim == 2 && direction isa EveryDirection)
                ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e],
                vgeo[ijk, _ξ2x2, e],
                vgeo[ijk, _ξ2x3, e]
            end

            @unroll for s in 1:ngradlapstate
                G1 = Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e]
                G2 = Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e]
                G3 = Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e]

                s_grad[1, i, j, s] = M * (ξ1x1 * G1 + ξ1x2 * G2 + ξ1x3 * G3)
                if dim == 3
                    s_grad[2, i, j, s] = M * (ξ2x1 * G1 + ξ2x2 * G2 + ξ2x3 * G3)
                end
            end
            @synchronize

            MI = local_MI[k]
            @unroll for s in 1:ngradlapstate
                @unroll for n in 1:Nq1
                    Dni = D[n, i]
                    local_div[k, s] -= MI * Dni * s_grad[1, n, j, s]
                    if dim == 3
                        Dnj = D[n, j]
                        local_div[k, s] -= MI * Dnj * s_grad[2, i, n, s]
                    end
                end
            end
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradlapstate
                if increment
                    Qhypervisc_div[ijk, s, e] += local_div[k, s]
                else
                    Qhypervisc_div[ijk, s, e] = local_div[k, s]
                end
            end
        end
    end
end

@kernel function volume_divergence_of_gradients!(
    balance_law::BalanceLaw,
    ::Val{info},
    ::VerticalDirection,
    Qhypervisc_grad,
    Qhypervisc_div,
    vgeo,
    D,
    elems,
    increment = false,
) where {info}
    @uniform begin
        dim = info.dim
        FT = eltype(Qhypervisc_grad)
        ngradlapstate = number_states(balance_law, GradientLaplacian())

        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        l_grad = MArray{Tuple{ngradlapstate}, FT}(undef)

        @inbounds Nqv = dim == 2 ? Nq2 : info.Nq[dim]
        s_grad_size = dim == 2 ? (Nq1, Nqv, ngradlapstate) : (0, 0, 0)
    end

    local_div = @private FT (Nq3, ngradlapstate)
    local_MI = @private FT (Nq3,)
    s_grad = @localmem FT s_grad_size

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            # initialize local tendency
            @unroll for s in 1:ngradlapstate
                local_div[k, s] = zero(FT)
            end
            # read in mass matrix inverse for element `e`
            local_MI[k] = vgeo[ijk, _MI, e]
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            M = vgeo[ijk, _M, e]

            if dim == 2
                ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e],
                vgeo[ijk, _ξ2x2, e],
                vgeo[ijk, _ξ2x3, e]
            else
                ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e],
                vgeo[ijk, _ξ3x2, e],
                vgeo[ijk, _ξ3x3, e]
            end

            @unroll for s in 1:ngradlapstate
                G1 = Qhypervisc_grad[ijk, 3 * (s - 1) + 1, e]
                G2 = Qhypervisc_grad[ijk, 3 * (s - 1) + 2, e]
                G3 = Qhypervisc_grad[ijk, 3 * (s - 1) + 3, e]

                if dim == 2
                    s_grad[i, j, s] = M * (ξ2x1 * G1 + ξ2x2 * G2 + ξ2x3 * G3)
                else
                    l_grad[s] = M * (ξ3x1 * G1 + ξ3x2 * G2 + ξ3x3 * G3)
                end
            end

            if dim == 3
                @unroll for n in 1:Nq3
                    MI = local_MI[n]
                    @unroll for s in 1:ngradlapstate
                        local_div[n, s] -= MI * D[k, n] * l_grad[s]
                    end
                end
            end

            @synchronize(dim == 2)

            if dim == 2
                MI = local_MI[k]
                @unroll for n in 1:Nqv
                    @unroll for s in 1:ngradlapstate
                        local_div[k, s] -= MI * D[n, j] * s_grad[i, n, s]
                    end
                end
            end
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradlapstate
                if increment
                    Qhypervisc_div[ijk, s, e] += local_div[k, s]
                else
                    Qhypervisc_div[ijk, s, e] = local_div[k, s]
                end
            end
        end
    end
end

"""
    function interface_divergence_of_gradients!(
        balance_law::BalanceLaw,
        ::Val{info},
        direction,
        divgradnumpenalty,
        Qhypervisc_grad,
        Qhypervisc_div,
        vgeo,
        sgeo,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        elems,
    )

Compute kernel for evaluating the interface divergence of gradients (or,
equivalently, the scalar laplacian) for the DG form:

∫ₑ ψ⋅ΔG dx - ∫ₑ ∇ψ⋅∇G dx + ∮ₑ n̂ ψ⋅(∇G)⋆ dS,

or equivalently in matrix form:

ΔG = M⁻¹(DᵀM ∇G + ∑ᶠ LᵀMf (∇G)⋆).

This kernel computes the interface terms: M⁻¹∑ᶠ LᵀMf (∇G)⋆
where M is the mass matrix, Mf is the face mass matrix, L is an interpolator
from volume to face, and (∇G)⋆ is the numerical fluxes for the gradients.
"""
@kernel function interface_divergence_of_gradients!(
    balance_law::BalanceLaw,
    ::Val{info},
    direction,
    divgradnumpenalty,
    Qhypervisc_grad,
    Qhypervisc_div,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
) where {info}
    @uniform begin
        dim = info.dim
        FT = eltype(Qhypervisc_grad)
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        nface = info.nface
        Np = info.Np
        Nqk = info.Nqk

        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        l_grad⁻ = MArray{Tuple{3, ngradlapstate}, FT}(undef)
        l_grad⁺ = MArray{Tuple{3, ngradlapstate}, FT}(undef)
        l_div = MArray{Tuple{ngradlapstate}, FT}(undef)
    end

    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    @inbounds for f in faces
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        bctag = elemtobndy[f, e⁻]
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        if bctag != 0
            # TODO: we will use vmap⁺ to store the boundary element info
            #be = e⁺
            #bcid = vid⁺
            e⁺ = e⁻
            vid⁺ = vid⁻
        end

        # Load minus side data
        @unroll for s in 1:ngradlapstate
            l_grad⁻[1, s] = Qhypervisc_grad[vid⁻, 3 * (s - 1) + 1, e⁻]
            l_grad⁻[2, s] = Qhypervisc_grad[vid⁻, 3 * (s - 1) + 2, e⁻]
            l_grad⁻[3, s] = Qhypervisc_grad[vid⁻, 3 * (s - 1) + 3, e⁻]
        end

        # Load plus side data
        @unroll for s in 1:ngradlapstate
            l_grad⁺[1, s] = Qhypervisc_grad[vid⁺, 3 * (s - 1) + 1, e⁺]
            l_grad⁺[2, s] = Qhypervisc_grad[vid⁺, 3 * (s - 1) + 2, e⁺]
            l_grad⁺[3, s] = Qhypervisc_grad[vid⁺, 3 * (s - 1) + 3, e⁺]
        end

        if bctag == 0
            numerical_flux_divergence!(
                divgradnumpenalty,
                balance_law,
                Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_div),
                normal_vector,
                Grad{vars_state(balance_law, GradientLaplacian(), FT)}(l_grad⁻),
                Grad{vars_state(balance_law, GradientLaplacian(), FT)}(l_grad⁺),
            )
        else
            bcs = boundary_conditions(balance_law)
            # TODO: there is probably a better way to unroll this loop
            Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
                bc = bcs[d]
                numerical_boundary_flux_divergence!(
                    divgradnumpenalty,
                    bc,
                    balance_law,
                    Vars{vars_state(balance_law, GradientLaplacian(), FT)}(
                        l_div,
                    ),
                    normal_vector,
                    Grad{vars_state(balance_law, GradientLaplacian(), FT)}(
                        l_grad⁻,
                    ),
                    Grad{vars_state(balance_law, GradientLaplacian(), FT)}(
                        l_grad⁺,
                    ),
                )
            end d -> throw(BoundsError(bcs, bctag))
        end

        @unroll for s in 1:ngradlapstate
            Qhypervisc_div[vid⁻, s, e⁻] += vMI * sM * l_div[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

"""
    function volume_gradients_of_laplacians!(
        balance_law::BalanceLaw,
        ::Val{info},
        direction,
        Qhypervisc_grad,
        Qhypervisc_div,
        state_prognostic,
        state_auxiliary,
        vgeo,
        ω,
        D,
        elems,
        t,
        increment = false,
    ) where {info}

Computes the volume integral for the auxiliary equation
(in DG strong form):

∫ₑ ψI⋅η dx = ∫ₑ ψI⋅∇ΔG dx + ∮ₑ nψI⋅((ΔG)⋆ - ΔG) dS,

or equivalently in matrix notation:

η = M⁻¹ LᵀMf((ΔG)⋆ - ΔG) + D ΔG

This kernel computes the volume gradient: D * ΔG, where
D is the differentiation matrix and ΔG is the laplacian
"""
@kernel function volume_gradients_of_laplacians!(
    balance_law::BalanceLaw,
    ::Val{info},
    direction,
    Qhypervisc_grad,
    Qhypervisc_div,
    state_prognostic,
    state_auxiliary,
    vgeo,
    ω,
    D,
    elems,
    t,
    increment = false,
) where {info}
    @uniform begin
        dim = info.dim

        FT = eltype(Qhypervisc_grad)
        num_state_prognostic = number_states(balance_law, Prognostic())
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        nhyperviscstate = number_states(balance_law, Hyperdiffusive())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        ngradtransformstate = num_state_prognostic

        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
    end

    s_lap = @localmem FT (Nq1, Nq2, ngradlapstate)
    local_state_prognostic = @private FT (ngradtransformstate, Nq3)
    local_state_auxiliary = @private FT (num_state_auxiliary, Nq3)
    l_grad_lap = @private FT (3, ngradlapstate, Nq3)
    lap_ξ3 = @private FT (ngradlapstate, Nq3)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds @views begin
        @unroll for k in 1:Nq3
            @unroll for s in 1:ngradlapstate
                l_grad_lap[1, s, k] = -zero(FT)
                l_grad_lap[2, s, k] = -zero(FT)
                l_grad_lap[3, s, k] = -zero(FT)
                lap_ξ3[s, k] = -zero(FT)
            end

            # Load prognostic and auxiliary variables
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradtransformstate
                local_state_prognostic[s, k] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s, k] = state_auxiliary[ijk, s, e]
            end
        end

        @unroll for k in 1:Nq3
            # store laplacian into shared memory
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradlapstate
                s_lap[i, j, s] = Qhypervisc_div[ijk, s, e]
            end
            @synchronize

            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            ξ1x1, ξ1x2, ξ1x3 =
                vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]

            # Compute gradient of each state
            @unroll for s in 1:ngradlapstate
                lap_ξ1 = lap_ξ2 = zero(FT)

                @unroll for n in 1:Nq1
                    lap_ξ1 += D[i, n] * s_lap[n, j, s]
                    if dim == 3 || (dim == 2 && direction isa EveryDirection)
                        lap_ξ2 += D[j, n] * s_lap[i, n, s]
                    end
                    if dim == 3 && direction isa EveryDirection
                        lap_ξ3[s, n] += D[n, k] * s_lap[i, j, s]
                    end
                end

                # Application of chain-rule in ξ1 and ξ2 directions,
                # ∂G/∂xi = ∂ξ1/∂xi * ∂G/∂ξ1, ∂G/∂xi = ∂ξ2/∂xi * ∂G/∂ξ2
                # to get a physical gradient
                l_grad_lap[1, s, k] = ξ1x1 * lap_ξ1
                l_grad_lap[2, s, k] = ξ1x2 * lap_ξ1
                l_grad_lap[3, s, k] = ξ1x3 * lap_ξ1

                if dim == 3 || (dim == 2 && direction isa EveryDirection)
                    ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e],
                    vgeo[ijk, _ξ2x2, e],
                    vgeo[ijk, _ξ2x3, e]
                    l_grad_lap[1, s, k] += ξ2x1 * lap_ξ2
                    l_grad_lap[2, s, k] += ξ2x2 * lap_ξ2
                    l_grad_lap[3, s, k] += ξ2x3 * lap_ξ2
                end

            end

            # Synchronize threads on the device
            @synchronize
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            # Application of chain-rule in ξ3-direction: ∂G/∂xi = ∂ξ3/∂xi * ∂G/∂ξ3
            if dim == 3 && direction isa EveryDirection
                ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e],
                vgeo[ijk, _ξ3x2, e],
                vgeo[ijk, _ξ3x3, e]
                l_grad_lap[1, s, k] += ξ3x1 * lap_ξ3[s, k]
                l_grad_lap[2, s, k] += ξ3x2 * lap_ξ3[s, k]
                l_grad_lap[3, s, k] += ξ3x3 * lap_ξ3[s, k]
            end

            fill!(
                local_state_hyperdiffusion,
                -zero(eltype(local_state_hyperdiffusion)),
            )

            # Applies a linear transformation of gradients to the hyperdiffusive variables
            transform_post_gradient_laplacian_arr!(
                balance_law,
                local_state_hyperdiffusion,
                l_grad_lap[:, :, k],
                local_state_prognostic[:, k],
                local_state_auxiliary[:, k],
                t,
            )

            # Write out the result of the kernel to global memory
            @unroll for s in 1:nhyperviscstate
                if increment
                    Qhypervisc_grad[ijk, s, e] += local_state_hyperdiffusion[s]
                else
                    Qhypervisc_grad[ijk, s, e] = local_state_hyperdiffusion[s]
                end
            end
        end
    end
end

@kernel function volume_gradients_of_laplacians!(
    balance_law::BalanceLaw,
    ::Val{info},
    ::VerticalDirection,
    Qhypervisc_grad,
    Qhypervisc_div,
    state_prognostic,
    state_auxiliary,
    vgeo,
    ω,
    D,
    elems,
    t,
    increment = false,
) where {info}
    @uniform begin
        dim = info.dim

        FT = eltype(Qhypervisc_grad)
        num_state_prognostic = number_states(balance_law, Prognostic())
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        nhyperviscstate = number_states(balance_law, Hyperdiffusive())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        ngradtransformstate = num_state_prognostic

        @inbounds Nq1 = info.Nq[1]
        @inbounds Nq2 = info.Nq[2]
        Nq3 = info.Nqk

        _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
        _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
        _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

        @inbounds Nqv = dim == 2 ? Nq2 : info.Nq[dim]
        lap_ζ_size = dim == 3 ? (ngradlapstate, Nq3) : (0, 0)
        shared_lap_dim2 = dim == 2 ? Nqv : Nq1

        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)
    end

    s_lap = @localmem FT (Nq1, shared_lap_dim2, ngradlapstate)
    local_state_prognostic = @private FT (ngradtransformstate, Nq3)
    local_state_auxiliary = @private FT (num_state_auxiliary, Nq3)
    l_grad_lap = @private FT (3, ngradlapstate, Nq3)

    local_ζ = @private FT (3, Nq3)

    lap_ζ = @private FT lap_ζ_size

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds @views begin
        @unroll for k in 1:Nq3
            @unroll for s in 1:ngradlapstate
                l_grad_lap[1, s, k] = -zero(FT)
                l_grad_lap[2, s, k] = -zero(FT)
                l_grad_lap[3, s, k] = -zero(FT)
                if dim == 3
                    lap_ζ[s, k] = -zero(FT)
                end
            end

            # Load prognostic and auxiliary variables
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradtransformstate
                local_state_prognostic[s, k] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s, k] = state_auxiliary[ijk, s, e]
            end

            # Load geometry terms for the Jacobian: ∂ζ/∂xⱼ
            local_ζ[1, k] = vgeo[ijk, _ζx1, e]
            local_ζ[2, k] = vgeo[ijk, _ζx2, e]
            local_ζ[3, k] = vgeo[ijk, _ζx3, e]
        end

        @unroll for k in 1:Nq3
            # store laplacian into shared memory
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            @unroll for s in 1:ngradlapstate
                s_lap[i, j, s] = Qhypervisc_div[ijk, s, e]
            end

            @synchronize

            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            # Compute gradient of each state
            @unroll for s in 1:ngradlapstate
                if dim == 2
                    lap_ζ1 = zero(FT)
                    @unroll for n in 1:Nqv
                        lap_ζ1 += D[j, n] * s_lap[i, n, s]
                    end
                    # Application of chain-rule in ζ direction
                    # ∂G/∂xi = ∂ζ/∂xi * ∂G/∂ζ
                    # to get a physical gradient
                    l_grad_lap[1, s, k] = local_ζ[1, k] * lap_ζ1
                    l_grad_lap[2, s, k] = local_ζ[2, k] * lap_ζ1
                    l_grad_lap[3, s, k] = local_ζ[3, k] * lap_ζ1
                else
                    @unroll for n in 1:Nq3
                        lap_ζ[s, n] += D[n, k] * s_lap[i, j, s]
                    end
                end
            end
            # Synchronize threads on the device
            @synchronize
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            # Application of chain-rule in ξ3-direction: ∂G/∂xi = ∂ξ3/∂xi * ∂G/∂ξ3
            if dim == 3
                ζx1, ζx2, ζx3 = local_ζ[1, k], local_ζ[2, k], local_ζ[3, k]
                @unroll for s in 1:ngradlapstate
                    l_grad_lap[1, s, k] += ζx1 * lap_ζ[s, k]
                    l_grad_lap[2, s, k] += ζx2 * lap_ζ[s, k]
                    l_grad_lap[3, s, k] += ζx3 * lap_ζ[s, k]
                end
            end

            fill!(
                local_state_hyperdiffusion,
                -zero(eltype(local_state_hyperdiffusion)),
            )

            # Applies a linear transformation of gradients to the hyperdiffusive variables
            transform_post_gradient_laplacian_arr!(
                balance_law,
                local_state_hyperdiffusion,
                l_grad_lap[:, :, k],
                local_state_prognostic[:, k],
                local_state_auxiliary[:, k],
                t,
            )

            # Write out the result of the kernel to global memory
            @unroll for s in 1:nhyperviscstate
                if increment
                    Qhypervisc_grad[ijk, s, e] += local_state_hyperdiffusion[s]
                else
                    Qhypervisc_grad[ijk, s, e] = local_state_hyperdiffusion[s]
                end
            end
        end
    end
end

"""
    function interface_gradients_of_laplacians!(
        balance_law::BalanceLaw,
        ::Val{info},
        direction,
        hyperviscnumflux,
        Qhypervisc_grad,
        Qhypervisc_div,
        state_prognostic,
        state_auxiliary,
        vgeo,
        sgeo,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        elems,
        t,
    )

Computes the volume integral for the auxiliary equation
(in DG strong form):

∫ₑ ψI⋅η dx = ∫ₑ ψI⋅∇ΔG dx + ∮ₑ nψI⋅((ΔG)⋆ - ΔG) dS,

or equivalently in matrix notation:

η = M⁻¹ LᵀMf((ΔG)⋆ - ΔG) + D ΔG

This kernel computes the interface gradient term: M⁻¹ LᵀMf((ΔG)⋆ - ΔG),
where M is the mass matrix, Mf is the face mass matrix, L is an interpolator
from volume to face, ΔG is the laplacian, and (ΔG)⋆ is
the associated numerical flux.
"""
@kernel function interface_gradients_of_laplacians!(
    balance_law::BalanceLaw,
    ::Val{info},
    direction,
    hyperviscnumflux,
    Qhypervisc_grad,
    Qhypervisc_div,
    state_prognostic,
    state_auxiliary,
    vgeo,
    sgeo,
    vmap⁻,
    vmap⁺,
    elemtobndy,
    elems,
    t,
) where {info}
    @uniform begin
        dim = info.dim
        FT = eltype(Qhypervisc_grad)
        num_state_prognostic = number_states(balance_law, Prognostic())
        ngradlapstate = number_states(balance_law, GradientLaplacian())
        nhyperviscstate = number_states(balance_law, Hyperdiffusive())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        ngradtransformstate = num_state_prognostic
        nface = info.nface
        Np = info.Np
        Nqk = info.Nqk

        faces = 1:nface
        if direction isa VerticalDirection
            faces = (nface - 1):nface
        elseif direction isa HorizontalDirection
            faces = 1:(nface - 2)
        end

        l_lap⁻ = MArray{Tuple{ngradlapstate}, FT}(undef)
        l_lap⁺ = MArray{Tuple{ngradlapstate}, FT}(undef)
        local_state_hyperdiffusion = MArray{Tuple{nhyperviscstate}, FT}(undef)

        local_state_prognostic⁻ = MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_prognostic⁺ = MArray{Tuple{ngradtransformstate}, FT}(undef)
        local_state_auxiliary⁺ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    end

    eI = @index(Group, Linear)
    n = @index(Local, Linear)

    e = @private Int (1,)
    @inbounds e[1] = elems[eI]

    @inbounds for f in faces
        e⁻ = e[1]
        normal_vector = SVector(
            sgeo[_n1, n, f, e⁻],
            sgeo[_n2, n, f, e⁻],
            sgeo[_n3, n, f, e⁻],
        )
        bctag = elemtobndy[f, e⁻]
        sM, vMI = sgeo[_sM, n, f, e⁻], sgeo[_vMI, n, f, e⁻]
        id⁻, id⁺ = vmap⁻[n, f, e⁻], vmap⁺[n, f, e⁻]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1

        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        if bctag != 0
            # TODO: we will use vmap⁺ to store the boundary element info
            #be = e⁺
            #bcid = vid⁺
            e⁺ = e⁻
            vid⁺ = vid⁻
        end

        # Load minus side data
        @unroll for s in 1:ngradtransformstate
            local_state_prognostic⁻[s] = state_prognostic[vid⁻, s, e⁻]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁻[s] = state_auxiliary[vid⁻, s, e⁻]
        end

        @unroll for s in 1:ngradlapstate
            l_lap⁻[s] = Qhypervisc_div[vid⁻, s, e⁻]
        end

        # Load plus side data
        @unroll for s in 1:ngradtransformstate
            local_state_prognostic⁺[s] = state_prognostic[vid⁺, s, e⁺]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary⁺[s] = state_auxiliary[vid⁺, s, e⁺]
        end

        @unroll for s in 1:ngradlapstate
            l_lap⁺[s] = Qhypervisc_div[vid⁺, s, e⁺]
        end

        if bctag == 0
            numerical_flux_higher_order!(
                hyperviscnumflux,
                balance_law,
                Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                    local_state_hyperdiffusion,
                ),
                normal_vector,
                Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_lap⁻),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁻,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁻,
                ),
                Vars{vars_state(balance_law, GradientLaplacian(), FT)}(l_lap⁺),
                Vars{vars_state(balance_law, Prognostic(), FT)}(
                    local_state_prognostic⁺,
                ),
                Vars{vars_state(balance_law, Auxiliary(), FT)}(
                    local_state_auxiliary⁺,
                ),
                t,
            )
        else
            bcs = boundary_conditions(balance_law)
            # TODO: there is probably a better way to unroll this loop
            Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
                bc = bcs[d]
                numerical_boundary_flux_higher_order!(
                    hyperviscnumflux,
                    bc,
                    balance_law,
                    Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                        local_state_hyperdiffusion,
                    ),
                    normal_vector,
                    Vars{vars_state(balance_law, GradientLaplacian(), FT)}(
                        l_lap⁻,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁻,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁻,
                    ),
                    Vars{vars_state(balance_law, GradientLaplacian(), FT)}(
                        l_lap⁺,
                    ),
                    Vars{vars_state(balance_law, Prognostic(), FT)}(
                        local_state_prognostic⁺,
                    ),
                    Vars{vars_state(balance_law, Auxiliary(), FT)}(
                        local_state_auxiliary⁺,
                    ),
                    t,
                )
            end d -> throw(BoundsError(bcs, bctag))
        end

        @unroll for s in 1:nhyperviscstate
            Qhypervisc_grad[vid⁻, s, e⁻] +=
                vMI * sM * local_state_hyperdiffusion[s]
        end
        # Need to wait after even faces to avoid race conditions
        @synchronize(f % 2 == 0)
    end
end

@kernel function kernel_local_courant!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    pointwise_courant,
    local_courant,
    state_prognostic,
    state_auxiliary,
    state_gradient_flux,
    elems,
    Δt,
    simtime,
    direction,
) where {dim, N}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_auxiliary = number_states(balance_law, Auxiliary())

        Nq = N .+ 1
        @inbounds Nqk = dim == 2 ? 1 : Nq[dim]
        @inbounds Np = Nq[1] * Nq[2] * Nqk

        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
    end

    I = @index(Global, Linear)
    e = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic[s] = state_prognostic[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux[s] = state_gradient_flux[n, s, e]
        end

        Δx = pointwise_courant[n, e]
        c = local_courant(
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                local_state_prognostic,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                local_state_auxiliary,
            ),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                local_state_gradient_flux,
            ),
            Δx,
            Δt,
            simtime,
            direction,
        )

        pointwise_courant[n, e] = c
    end
end

@kernel function dgsem_auxiliary_field_gradient!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{N},
    direction,
    ∇state,
    state,
    vgeo,
    D,
    ω,
    ::Val{I},
    ::Val{O},
    increment,
) where {dim, N, I, O}
    @uniform begin
        FT = eltype(state)
        ngradstate = length(I)
        Nq = N .+ 1
        @inbounds begin
            Nq1 = Nq[1]
            Nq2 = Nq[2]
            Nq3 = dim == 2 ? 1 : Nq[dim]
        end
    end

    shared_state = @localmem FT (Nq1, Nq2, ngradstate)

    local_gradient = @private FT (3, ngradstate, Nq3)
    Gξ3 = @private FT (ngradstate, Nq3)

    e = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds @views begin
        @unroll for k in 1:Nq3
            @unroll for s in 1:ngradstate
                local_gradient[1, s, k] = -zero(FT)
                local_gradient[2, s, k] = -zero(FT)
                local_gradient[3, s, k] = -zero(FT)
                Gξ3[s, k] = -zero(FT)
            end
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            @unroll for s in 1:ngradstate
                shared_state[i, j, s] = state[ijk, I[s], e]
            end
            @synchronize

            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
            ξ1x1, ξ1x2, ξ1x3 =
                vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]

            # Compute gradient of each state
            @unroll for s in 1:ngradstate
                Gξ1 = Gξ2 = zero(FT)

                if (dim == 2 && (direction isa VerticalDirection))
                    @unroll for n in 1:Nq2
                        Gξ2 += D[j, n] * shared_state[i, n, s]
                    end
                end
                if (dim == 3 && (direction isa VerticalDirection))
                    @unroll for n in 1:Nq3
                        Gξ3[s, n] += D[n, k] * shared_state[i, j, s]
                    end
                end

                if (dim == 2 && (direction isa HorizontalDirection))
                    @unroll for n in 1:Nq1
                        Gξ1 += D[i, n] * shared_state[n, j, s]
                    end
                end
                if (dim == 3 && (direction isa HorizontalDirection))
                    @unroll for n in 1:Nq1
                        Gξ1 += D[i, n] * shared_state[n, j, s]
                        Gξ2 += D[j, n] * shared_state[i, n, s]
                    end
                end

                if (direction isa HorizontalDirection)
                    local_gradient[1, s, k] += ξ1x1 * Gξ1
                    local_gradient[2, s, k] += ξ1x2 * Gξ1
                    local_gradient[3, s, k] += ξ1x3 * Gξ1
                end

                if (
                    (dim == 2 && (direction isa VerticalDirection)) ||
                    (dim == 3 && (direction isa HorizontalDirection))
                )
                    ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e],
                    vgeo[ijk, _ξ2x2, e],
                    vgeo[ijk, _ξ2x3, e]
                    local_gradient[1, s, k] += ξ2x1 * Gξ2
                    local_gradient[2, s, k] += ξ2x2 * Gξ2
                    local_gradient[3, s, k] += ξ2x3 * Gξ2
                end
            end
            @synchronize
        end

        @unroll for k in 1:Nq3
            ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

            if (dim == 3 && (direction isa VerticalDirection))
                ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e],
                vgeo[ijk, _ξ3x2, e],
                vgeo[ijk, _ξ3x3, e]
                @unroll for s in 1:ngradstate
                    local_gradient[1, s, k] += ξ3x1 * Gξ3[s, k]
                    local_gradient[2, s, k] += ξ3x2 * Gξ3[s, k]
                    local_gradient[3, s, k] += ξ3x3 * Gξ3[s, k]
                end
            end

            if increment
                @unroll for s in 1:ngradstate
                    ∇state[ijk, O[3 * (s - 1) + 1], e] +=
                        local_gradient[1, s, k]
                    ∇state[ijk, O[3 * (s - 1) + 2], e] +=
                        local_gradient[2, s, k]
                    ∇state[ijk, O[3 * (s - 1) + 3], e] +=
                        local_gradient[3, s, k]
                end
            else
                @unroll for s in 1:ngradstate
                    ∇state[ijk, O[3 * (s - 1) + 1], e] = local_gradient[1, s, k]
                    ∇state[ijk, O[3 * (s - 1) + 2], e] = local_gradient[2, s, k]
                    ∇state[ijk, O[3 * (s - 1) + 3], e] = local_gradient[3, s, k]
                end
            end
        end
    end
end
