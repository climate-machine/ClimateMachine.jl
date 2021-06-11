using DocStringExtensions
using KernelAbstractions
using Adapt

using ..Mesh.Geometry
using ..VariableTemplates

import ..Mesh.Grids:
    _ξ1x1, _ξ2x1, _ξ3x1, _ξ1x2, _ξ2x2, _ξ3x2, _ξ1x3, _ξ2x3, _ξ3x3, tpxv!, ftpxv!
import ..MPIStateArrays: array_device

"""
    VectorGradients{
        FT <: AbstractFloat,
        FTA2D <: AbstractArray{FT, 2},
        FTA3D <: AbstractArray{FT, 3},
    }
This data structure stores the spatial gradients of a velocity field.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct VectorGradients{FT <: AbstractFloat, FTA2D <: AbstractArray{FT, 2}}
    "∂u₁/∂x₁"
    ∂₁u₁::FTA2D
    "∂u₁/∂x₂"
    ∂₂u₁::FTA2D
    "∂u₁/∂x₃"
    ∂₃u₁::FTA2D
    "∂u₂/∂x₁"
    ∂₁u₂::FTA2D
    "∂u₂/∂x₂"
    ∂₂u₂::FTA2D
    "∂u₂/∂x₃"
    ∂₃u₂::FTA2D
    "∂u₃/∂x₁"
    ∂₁u₃::FTA2D
    "∂u₃/∂x₂"
    ∂₂u₃::FTA2D
    "∂u₃/∂x₃"
    ∂₃u₃::FTA2D
end
Adapt.@adapt_structure VectorGradients

function VectorGradients(
    npts::Int,
    nel::Int,
    ::Type{FT},
    ::Type{DA},
) where {FT <: AbstractFloat, DA}

    return VectorGradients{FT, DA{FT, 2}}(
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
        DA{FT, 2}(undef, npts, nel),
    )
end

"""
    VectorGradients(dg::SpaceDiscretization, Q::MPIStateArray)
This constructor computes the spatial gradients of the velocity field.
# Arguments
 - `dg`: SpaceDiscretization
 - `Q`: MPIStateArray containing the prognostic state variables
"""
function VectorGradients(dg::SpaceDiscretization, Q::MPIStateArray)
    bl = dg.balance_law
    FT = eltype(dg.grid)
    Nq = polynomialorders(dg.grid) .+ 1 #N + 1
    Nqmax = maximum(Nq)
    npoints = prod(Nq)
    realelems = dg.grid.topology.realelems
    nrealelem = length(realelems)

    DA = (typeof(parent(Q.data)) <: Array) ? Array : CuArray
    device = DA == Array ? CPU() : CUDADevice()
    vgrad = VectorGradients(npoints, nrealelem, FT, DA)
    ind = varsindices(vars_state(bl, Prognostic(), FT), ("ρ", "ρu"))
    _ρ, _ρu, _ρv, _ρw = ind[1], ind[2], ind[3], ind[4]

    ρ = view(Q.data, :, _ρ, realelems)
    ρu = view(Q.data, :, _ρu, realelems)
    ρv = view(Q.data, :, _ρv, realelems)
    ρw = view(Q.data, :, _ρw, realelems)

    ξ1x1 = view(dg.grid.vgeo, :, _ξ1x1, :)
    ξ1x2 = view(dg.grid.vgeo, :, _ξ1x2, :)
    ξ1x3 = view(dg.grid.vgeo, :, _ξ1x3, :)
    ξ2x1 = view(dg.grid.vgeo, :, _ξ2x1, :)
    ξ2x2 = view(dg.grid.vgeo, :, _ξ2x2, :)
    ξ2x3 = view(dg.grid.vgeo, :, _ξ2x3, :)
    ξ3x1 = view(dg.grid.vgeo, :, _ξ3x1, :)
    ξ3x2 = view(dg.grid.vgeo, :, _ξ3x2, :)
    ξ3x3 = view(dg.grid.vgeo, :, _ξ3x3, :)

    ∂f1∂ξ₁ = view(dg.grid.scratch, :, 1, :)
    ∂f1∂ξ₂ = view(dg.grid.scratch, :, 2, :)
    ∂f1∂ξ₃ = view(dg.grid.scratch, :, 3, :)

    ∂f2∂ξ₁ = view(dg.grid.scratch, :, 4, :)
    ∂f2∂ξ₂ = view(dg.grid.scratch, :, 5, :)
    ∂f2∂ξ₃ = view(dg.grid.scratch, :, 6, :)

    ∂f3∂ξ₁ = view(dg.grid.scratch, :, 7, :)
    ∂f3∂ξ₂ = view(dg.grid.scratch, :, 8, :)
    ∂f3∂ξ₃ = view(dg.grid.scratch, :, 9, :)

    @show "timing tpxv! kernels"
    for i in 1:10
        @time begin
            tpxv!(dg.grid, :∂ξ₁, false, ρu, ∂f1∂ξ₁, vin_den = ρ) # ∂u₁/∂ξ₁
            tpxv!(dg.grid, :∂ξ₂, false, ρu, ∂f1∂ξ₂, vin_den = ρ) # ∂u₁/∂ξ₂
            tpxv!(dg.grid, :∂ξ₃, false, ρu, ∂f1∂ξ₃, vin_den = ρ) # ∂u₁/∂ξ₃
        end
        println("-------------")
    end
    vgrad.∂₁u₁ .= ∂f1∂ξ₁ .* ξ1x1 .+ ∂f1∂ξ₂ .* ξ2x1 .+ ∂f1∂ξ₃ .* ξ3x1 # ∂u₁/∂x₁
    vgrad.∂₂u₁ .= ∂f1∂ξ₁ .* ξ1x2 .+ ∂f1∂ξ₂ .* ξ2x2 .+ ∂f1∂ξ₃ .* ξ3x2 # ∂u₁/∂x₂
    vgrad.∂₃u₁ .= ∂f1∂ξ₁ .* ξ1x3 .+ ∂f1∂ξ₂ .* ξ2x3 .+ ∂f1∂ξ₃ .* ξ3x3 # ∂u₁/∂x₃

    tpxv!(dg.grid, :∂ξ₁, false, ρv, ∂f2∂ξ₁, vin_den = ρ) # ∂u₂/∂ξ₁
    tpxv!(dg.grid, :∂ξ₂, false, ρv, ∂f2∂ξ₂, vin_den = ρ) # ∂u₂/∂ξ₂
    tpxv!(dg.grid, :∂ξ₃, false, ρv, ∂f2∂ξ₃, vin_den = ρ) # ∂u₂/∂ξ₃

    vgrad.∂₁u₂ .= ∂f2∂ξ₁ .* ξ1x1 .+ ∂f2∂ξ₂ .* ξ2x1 .+ ∂f2∂ξ₃ .* ξ3x1 # ∂u₂/∂x₁
    vgrad.∂₂u₂ .= ∂f2∂ξ₁ .* ξ1x2 .+ ∂f2∂ξ₂ .* ξ2x2 .+ ∂f2∂ξ₃ .* ξ3x2 # ∂u₂/∂x₂
    vgrad.∂₃u₂ .= ∂f2∂ξ₁ .* ξ1x3 .+ ∂f2∂ξ₂ .* ξ2x3 .+ ∂f2∂ξ₃ .* ξ3x3 # ∂u₂/∂x₃

    tpxv!(dg.grid, :∂ξ₁, false, ρw, ∂f3∂ξ₁, vin_den = ρ) # ∂u₃/∂ξ₁
    tpxv!(dg.grid, :∂ξ₂, false, ρw, ∂f3∂ξ₂, vin_den = ρ) # ∂u₃/∂ξ₂
    tpxv!(dg.grid, :∂ξ₃, false, ρw, ∂f3∂ξ₃, vin_den = ρ) # ∂u₃/∂ξ₃

    vgrad.∂₁u₃ .= ∂f3∂ξ₁ .* ξ1x1 .+ ∂f3∂ξ₂ .* ξ2x1 .+ ∂f3∂ξ₃ .* ξ3x1 # ∂u₃/∂x₁
    vgrad.∂₂u₃ .= ∂f3∂ξ₁ .* ξ1x2 .+ ∂f3∂ξ₂ .* ξ2x2 .+ ∂f3∂ξ₃ .* ξ3x2 # ∂u₃/∂x₂
    vgrad.∂₃u₃ .= ∂f3∂ξ₁ .* ξ1x3 .+ ∂f3∂ξ₂ .* ξ2x3 .+ ∂f3∂ξ₃ .* ξ3x3 # ∂u₃/∂x₃
    #        end
    #    end

    @show "timing ftpxv! kernels"
    for i in 1:10
        @time begin
            ftpxv!(dg.grid, :∂ξ₁, false, ρu, ∂f1∂ξ₁, vin_den = ρ) # ∂u₁/∂ξ₁
            ftpxv!(dg.grid, :∂ξ₂, false, ρu, ∂f1∂ξ₂, vin_den = ρ) # ∂u₁/∂ξ₂
            ftpxv!(dg.grid, :∂ξ₃, false, ρu, ∂f1∂ξ₃, vin_den = ρ) # ∂u₁/∂ξ₃
        end
        println("-------------")
    end
    vgrad.∂₁u₁ .= ∂f1∂ξ₁ .* ξ1x1 .+ ∂f1∂ξ₂ .* ξ2x1 .+ ∂f1∂ξ₃ .* ξ3x1 # ∂u₁/∂x₁
    vgrad.∂₂u₁ .= ∂f1∂ξ₁ .* ξ1x2 .+ ∂f1∂ξ₂ .* ξ2x2 .+ ∂f1∂ξ₃ .* ξ3x2 # ∂u₁/∂x₂
    vgrad.∂₃u₁ .= ∂f1∂ξ₁ .* ξ1x3 .+ ∂f1∂ξ₂ .* ξ2x3 .+ ∂f1∂ξ₃ .* ξ3x3 # ∂u₁/∂x₃

    ftpxv!(dg.grid, :∂ξ₁, false, ρv, ∂f2∂ξ₁, vin_den = ρ) # ∂u₂/∂ξ₁
    ftpxv!(dg.grid, :∂ξ₂, false, ρv, ∂f2∂ξ₂, vin_den = ρ) # ∂u₂/∂ξ₂
    ftpxv!(dg.grid, :∂ξ₃, false, ρv, ∂f2∂ξ₃, vin_den = ρ) # ∂u₂/∂ξ₃

    vgrad.∂₁u₂ .= ∂f2∂ξ₁ .* ξ1x1 .+ ∂f2∂ξ₂ .* ξ2x1 .+ ∂f2∂ξ₃ .* ξ3x1 # ∂u₂/∂x₁
    vgrad.∂₂u₂ .= ∂f2∂ξ₁ .* ξ1x2 .+ ∂f2∂ξ₂ .* ξ2x2 .+ ∂f2∂ξ₃ .* ξ3x2 # ∂u₂/∂x₂
    vgrad.∂₃u₂ .= ∂f2∂ξ₁ .* ξ1x3 .+ ∂f2∂ξ₂ .* ξ2x3 .+ ∂f2∂ξ₃ .* ξ3x3 # ∂u₂/∂x₃

    ftpxv!(dg.grid, :∂ξ₁, false, ρw, ∂f3∂ξ₁, vin_den = ρ) # ∂u₃/∂ξ₁
    ftpxv!(dg.grid, :∂ξ₂, false, ρw, ∂f3∂ξ₂, vin_den = ρ) # ∂u₃/∂ξ₂
    ftpxv!(dg.grid, :∂ξ₃, false, ρw, ∂f3∂ξ₃, vin_den = ρ) # ∂u₃/∂ξ₃

    vgrad.∂₁u₃ .= ∂f3∂ξ₁ .* ξ1x1 .+ ∂f3∂ξ₂ .* ξ2x1 .+ ∂f3∂ξ₃ .* ξ3x1 # ∂u₃/∂x₁
    vgrad.∂₂u₃ .= ∂f3∂ξ₁ .* ξ1x2 .+ ∂f3∂ξ₂ .* ξ2x2 .+ ∂f3∂ξ₃ .* ξ3x2 # ∂u₃/∂x₂
    vgrad.∂₃u₃ .= ∂f3∂ξ₁ .* ξ1x3 .+ ∂f3∂ξ₂ .* ξ2x3 .+ ∂f3∂ξ₃ .* ξ3x3 # ∂u₃/∂x₃
    #        end
    #    end

    return vgrad
end
#--------------------------------------------------------------------------------------------------

"""
    Vorticity{
        FT <: AbstractFloat,
        FTA2D <: AbstractArray{FT, 2},
        FTA3D <: AbstractArray{FT, 3},
    }
This data structure stores the vorticity of a velocity field.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct Vorticity{FT <: AbstractFloat, FTA2D <: AbstractArray{FT, 2}}
    "x1 component of vorticity"
    Ω₁::FTA2D
    "x2 component of vorticity"
    Ω₂::FTA2D
    "x3 component of vorticity"
    Ω₃::FTA2D
end
Adapt.@adapt_structure Vorticity

function Vorticity(
    npoints::Int,
    nel::Int,
    ::Type{FT},
    ::Type{DA},
) where {FT <: AbstractFloat, DA}
    return Vorticity{FT, DA{FT, 2}}(
        DA{FT, 2}(undef, npoints, nel),
        DA{FT, 2}(undef, npoints, nel),
        DA{FT, 2}(undef, npoints, nel),
    )
end

"""
    Vorticity(
        dg::SpaceDiscretization,
        vgrad::VectorGradients,
    )
This function computes the vorticity of the velocity field.
# Arguments
 - `dg`: SpaceDiscretization
 - `vgrad`: vector gradients
"""
function Vorticity(dg::SpaceDiscretization, vgrad::VectorGradients)
    bl = dg.balance_law
    FT = eltype(dg.grid)
    DA = (typeof(vgrad.∂₁u₁) <: Array) ? Array : CuArray
    npoints = prod(polynomialorders(dg.grid) .+ 1)
    nrealelem = length(dg.grid.topology.realelems)

    vort = Vorticity(npoints, nrealelem, FT, DA)

    Ω₁, Ω₂, Ω₃ = 1, 2, 3

    vort.Ω₁ .= vgrad.∂₂u₃ .- vgrad.∂₃u₂
    vort.Ω₂ .= vgrad.∂₃u₁ .- vgrad.∂₁u₃
    vort.Ω₃ .= vgrad.∂₁u₂ .- vgrad.∂₂u₁

    return vort
end
