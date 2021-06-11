using DocStringExtensions
using KernelAbstractions
using CUDA

include("TensorProductKernels.jl")

"""
    tpxv!(grd::AbstractGrid{FT, 3},
        opts::Symbol,
        tropts::Bool,
        vin::AbstractArray{FT},
        vout::AbstractArray{FT},
        vin_den::Union{AbstractArray{FT}, Nothing} = nothing,
    ) where {FT<:AbstractFloat}

This function computes the inner derivatives of a field using the fast
tensor-product algorithm on a given mesh. Currently, DG mesh (`:m1`) and
over-integration mesh (`:m2`) are supported. The function provides the 
building blocks for constructing various terms in the DG formulation and 
for constructing some diagnostic fields.

# Arguments
 - `grd`: AbstractGrid
 - `opts`: inner derivative to be computed (currently, `:ξ1`, `:ξ2`, `:ξ3` supported)
 - `tropts`: transpose (true/false)
 - `vin`: input
 - `vout`: output
 - `vin_den`: Scale vin with vin_den
"""
function tpxv!(
    grd::AbstractGrid{FT, 3},
    opts::Symbol,
    tropts::Bool,
    vin::AbstractArray{FT},
    vout::AbstractArray{FT};
    vin_den::Union{AbstractArray{FT}, Nothing} = nothing,
    max_threads = 256,
) where {FT <: AbstractFloat}
    if grd isa DiscontinuousSpectralElementGrid # Spectral element mesh
        #---------------------------------------------------
        si, sj, sk = polynomialorders(grd) .+ 1
        sr, ss, st = si, sj, sk
        dims = (si, sj, sk, sr, ss, st)
        if opts == :∂ξ₁
            phir = (tropts ? grd.Dᵀ[1] : grd.D[1])
            phis, phit = nothing, nothing
        elseif opts == :∂ξ₂
            phis = (tropts ? grd.Dᵀ[2] : grd.D[2])
            phir, phit = nothing, nothing
        elseif opts == :∂ξ₃
            phit = (tropts ? grd.Dᵀ[3] : grd.D[3])
            phir, phis = nothing, nothing
        else
            error("tpxv!, DiscontinuousSpectralElementGrid: Unsupported opts")
        end
        #---------------------------------------------------
    else
        error("unsupported mesh/quadrature")
    end
    tpxv!(vin, vin_den, vout, phir, phis, phit, Val(dims), max_threads)
    return nothing
end
