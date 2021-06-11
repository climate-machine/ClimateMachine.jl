using DocStringExtensions
using KernelAbstractions
using CUDA

include("FastTensorProductKernels.jl")

"""
    ftpxv!(grd::DiscontinuousSpectralElementGrid,
        opts::Symbol,
        tropts::Bool,
        vin::AbstractArray{FT},
        vout::AbstractArray{FT},
        vin_den::Union{AbstractArray{FT}, Nothing} = nothing,
    ) where {FT<:AbstractFloat}
This function computes the inner derivatives of a field using the fast
tensor-product algorithm on a given mesh. Currently, DG mesh and
over-integration mesh (Quadrature mesh) are supported. The function provides the 
building blocks for constructing various terms in the DG formulation and 
for constructing some diagnostic fields.
# Arguments
 - `grd`: DiscontinuousSpectralElementGrid
 - `opts`: inner derivative to be computed (currently, `:ξ1`, `:ξ2`, `:ξ3` supported)
 - `tropts`: transpose (true/false)
 - `vin`: input
 - `vout`: output
 - `vin_den`: Scale vin with vin_den
"""
function ftpxv!(
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
            error("ftpxv!, DiscontinuousSpectralElementGrid: Unsupported opts")
        end
        #---------------------------------------------------
    elseif grd isa QuadratureGrid # over-integration grid
        if opts == :∂ξ₁
            phir = (tropts ? grd.Dᵀ[1] : grd.D[1])
            phis = (tropts ? grd.Bᵀ[2] : grd.B[2])
            phit = (tropts ? grd.Bᵀ[3] : grd.B[3])
        elseif opts == :∂ξ₂
            phir = (tropts ? grd.Bᵀ[1] : grd.B[1])
            phis = (tropts ? grd.Dᵀ[2] : grd.D[2])
            phit = (tropts ? grd.Bᵀ[3] : grd.B[3])
        elseif opts == :∂ξ₃
            phir = (tropts ? grd.Bᵀ[1] : grd.B[1])
            phis = (tropts ? grd.Bᵀ[2] : grd.B[2])
            phit = (tropts ? grd.Dᵀ[3] : grd.D[3])
        else
            error("ftpxv!, QuadratureGrid: Unsupported opts")
        end
        sr, si = size(phir)
        ss, sj = size(phis)
        st, sk = size(phit)
    else
        error("unsupported mesh/quadrature")
    end
    dims = (si, sj, sk, sr, ss, st)
    ftpxv!(
        vin,
        vin_den,
        vout,
        phir,
        phis,
        phit,
        Val(dims),
        grd.scratch_ftp,
        max_threads,
    )
    return nothing
end
