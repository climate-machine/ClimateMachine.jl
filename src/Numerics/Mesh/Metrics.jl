module Metrics

using ..GeometricFactors

export creategrid, compute_reference_to_physical_coord_jacobian, computemetric

"""
    creategrid!(vgeo, elemtocoord, ξ)

Create a 1-D grid using `elemtocoord` (see `brickmesh`) using the 1-D
`(-1, 1)` reference coordinates `ξ` (in 1D, `ξ = ξ1`). The element grids
are filled using linear interpolation of the element coordinates.

If `Nq = length(ξ)` and `nelem = size(elemtocoord, 3)` then the preallocated
array `vgeo.x1` should be `Nq * nelem == length(x1)`.
"""
function creategrid!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    e2c,
    ξ::NTuple{1, Vector{FT}},
) where {Nq, FT}
    (d, nvert, nelem) = size(e2c)
    @assert d == 1
    (ξ1,) = ξ
    x1 = reshape(vgeo.x1, (Nq..., nelem))

    # Linear blend
    @inbounds for e in 1:nelem
        for i in 1:Nq[1]
            vgeo.x1[i, e] =
                ((1 - ξ1[i]) * e2c[1, 1, e] + (1 + ξ1[i]) * e2c[1, 2, e]) / 2
        end
    end
    nothing
end

"""
    creategrid!(vgeo, elemtocoord, ξ)

Create a 2-D tensor product grid using `elemtocoord` (see `brickmesh`)
using the tuple `ξ = (ξ1, ξ2)`, composed by the 1D reference coordinates `ξ1` and `ξ2` in `(-1, 1)^2`.
The element grids are filled using bilinear interpolation of the element coordinates.

If `Nq = (length(ξ1), length(ξ2))` and `nelem = size(elemtocoord, 3)` then the
preallocated arrays `vgeo.x1` and `vgeo.x2` should be
`prod(Nq) * nelem == size(vgeo.x1) == size(vgeo.x2)`.
"""
function creategrid!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    e2c,
    ξ::NTuple{2, Vector{FT}},
) where {Nq, FT}
    (d, nvert, nelem) = size(e2c)
    @assert d == 2
    (ξ1, ξ2) = ξ
    x1 = reshape(vgeo.x1, (Nq..., nelem))
    x2 = reshape(vgeo.x2, (Nq..., nelem))

    # Bilinear blend of corners
    @inbounds for (f, n) in zip((x1, x2), 1:d)
        for e in 1:nelem, j in 1:Nq[2], i in 1:Nq[1]
            f[i, j, e] =
                (
                    (1 - ξ1[i]) * (1 - ξ2[j]) * e2c[n, 1, e] +
                    (1 + ξ1[i]) * (1 - ξ2[j]) * e2c[n, 2, e] +
                    (1 - ξ1[i]) * (1 + ξ2[j]) * e2c[n, 3, e] +
                    (1 + ξ1[i]) * (1 + ξ2[j]) * e2c[n, 4, e]
                ) / 4
        end
    end
    nothing
end

"""
    creategrid!(vgeo, elemtocoord, ξ)

Create a 3-D tensor product grid using `elemtocoord` (see `brickmesh`)
using the tuple `ξ = (ξ1, ξ2, ξ3)`, composed by the 1D reference coordinates `ξ1`, `ξ2`, `ξ3` in `(-1, 1)^3`.
The element grids are filled using trilinear interpolation of the element coordinates.

If `Nq = (length(ξ1), length(ξ2), length(ξ3))` and
`nelem = size(elemtocoord, 3)` then the preallocated arrays `vgeo.x1`, `vgeo.x2`,
and `vgeo.x3` should be `prod(Nq) * nelem == size(vgeo.x1) == size(vgeo.x2) == size(vgeo.x3)`.
"""
function creategrid!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    e2c,
    ξ::NTuple{3, Vector{FT}},
) where {Nq, FT}
    (d, nvert, nelem) = size(e2c)
    @assert d == 3
    (ξ1, ξ2, ξ3) = ξ
    x1 = reshape(vgeo.x1, (Nq..., nelem))
    x2 = reshape(vgeo.x2, (Nq..., nelem))
    x3 = reshape(vgeo.x3, (Nq..., nelem))

    # Trilinear blend of corners
    @inbounds for (f, n) in zip((x1, x2, x3), 1:d)
        for e in 1:nelem, k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]
            f[i, j, k, e] =
                (
                    (1 - ξ1[i]) * (1 - ξ2[j]) * (1 - ξ3[k]) * e2c[n, 1, e] +
                    (1 + ξ1[i]) * (1 - ξ2[j]) * (1 - ξ3[k]) * e2c[n, 2, e] +
                    (1 - ξ1[i]) * (1 + ξ2[j]) * (1 - ξ3[k]) * e2c[n, 3, e] +
                    (1 + ξ1[i]) * (1 + ξ2[j]) * (1 - ξ3[k]) * e2c[n, 4, e] +
                    (1 - ξ1[i]) * (1 - ξ2[j]) * (1 + ξ3[k]) * e2c[n, 5, e] +
                    (1 + ξ1[i]) * (1 - ξ2[j]) * (1 + ξ3[k]) * e2c[n, 6, e] +
                    (1 - ξ1[i]) * (1 + ξ2[j]) * (1 + ξ3[k]) * e2c[n, 7, e] +
                    (1 + ξ1[i]) * (1 + ξ2[j]) * (1 + ξ3[k]) * e2c[n, 8, e]
                ) / 8
        end
    end
    nothing
end

"""
    compute_reference_to_physical_coord_jacobian!(vgeo, nelem, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D::NTuple{2,Int}, a tuple of derivative matrices, i.e., D = (D1,), where:
    - D1::DAT2, 1-D derivative operator on the device in the first dimension

Compute the Jacobian matrix, ∂x / ∂ξ, of the transformation from reference coordinates,
`ξ1`, to physical coordinates, `vgeo.x1`, for each quadrature point in element e.
"""
function compute_reference_to_physical_coord_jacobian!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    nelem,
    D::NTuple{1, Matrix{FT}},
) where {Nq, FT}

    @assert Nq == map(d -> size(d, 1), D)

    T = eltype(vgeo.x1)
    (D1,) = D

    vgeo.x1ξ1 .= zero(T)

    for e in 1:nelem
        for i in 1:Nq[1]
            for n in 1:Nq[1]
                vgeo.x1ξ1[i, e] += D1[i, n] * vgeo.x1[n, e]
            end
        end
    end

    return vgeo
end

"""
    compute_reference_to_physical_coord_jacobian!(vgeo, nelem, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D::NTuple{2,Int}, a tuple of derivative matrices, i.e., D = (D1, D2), where:
    - D1::DAT2, 1-D derivative operator on the device in the first dimension
    - D2::DAT2, 1-D derivative operator on the device in the second dimension

Compute the Jacobian matrix, ∂x / ∂ξ, of the transformation from reference coordinates,
`ξ1`, `ξ2`, to physical coordinates, `vgeo.x1`, `vgeo.x2`,
for each quadrature point in element e.
"""
function compute_reference_to_physical_coord_jacobian!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    nelem,
    D::NTuple{2, Matrix{FT}},
) where {Nq, FT}

    @assert Nq == map(d -> size(d, 1), D)

    T = eltype(vgeo.x1)
    (D1, D2) = D

    x1 = reshape(vgeo.x1, (Nq..., nelem))
    x2 = reshape(vgeo.x2, (Nq..., nelem))
    x1ξ1 = reshape(vgeo.x1ξ1, (Nq..., nelem))
    x2ξ1 = reshape(vgeo.x2ξ1, (Nq..., nelem))
    x1ξ2 = reshape(vgeo.x1ξ2, (Nq..., nelem))
    x2ξ2 = reshape(vgeo.x2ξ2, (Nq..., nelem))

    x1ξ1 .= x1ξ2 .= zero(T)
    x2ξ1 .= x2ξ2 .= zero(T)

    for e in 1:nelem
        for j in 1:Nq[2], i in 1:Nq[1]
            for n in 1:Nq[1]
                x1ξ1[i, j, e] += D1[i, n] * x1[n, j, e]
                x2ξ1[i, j, e] += D1[i, n] * x2[n, j, e]
            end
            for n in 1:Nq[2]
                x1ξ2[i, j, e] += D2[j, n] * x1[i, n, e]
                x2ξ2[i, j, e] += D2[j, n] * x2[i, n, e]
            end
        end
    end

    return vgeo
end

"""
    compute_reference_to_physical_coord_jacobian!(vgeo, nelem, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- D::NTuple{3,Int}, a tuple of derivative matrices, i.e., D = (D1, D2, D3), where:
    - D1::DAT2, 1-D derivative operator on the device in the first dimension
    - D2::DAT2, 1-D derivative operator on the device in the second dimension
    - D3::DAT2, 1-D derivative operator on the device in the third dimension

Compute the Jacobian matrix, ∂x / ∂ξ, of the transformation from reference coordinates,
`ξ1`, `ξ2`, `ξ3` to physical coordinates, `vgeo.x1`, `vgeo.x2`, `vgeo.x3` for
each quadrature point in element e.
"""
function compute_reference_to_physical_coord_jacobian!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    nelem,
    D::NTuple{3, Matrix{FT}},
) where {Nq, FT}

    @assert Nq == map(d -> size(d, 1), D)

    T = eltype(vgeo.x1)
    (D1, D2, D3) = D

    x1 = reshape(vgeo.x1, (Nq..., nelem))
    x2 = reshape(vgeo.x2, (Nq..., nelem))
    x3 = reshape(vgeo.x3, (Nq..., nelem))
    x1ξ1 = reshape(vgeo.x1ξ1, (Nq..., nelem))
    x2ξ1 = reshape(vgeo.x2ξ1, (Nq..., nelem))
    x3ξ1 = reshape(vgeo.x3ξ1, (Nq..., nelem))
    x1ξ2 = reshape(vgeo.x1ξ2, (Nq..., nelem))
    x2ξ2 = reshape(vgeo.x2ξ2, (Nq..., nelem))
    x3ξ2 = reshape(vgeo.x3ξ2, (Nq..., nelem))
    x1ξ3 = reshape(vgeo.x1ξ3, (Nq..., nelem))
    x2ξ3 = reshape(vgeo.x2ξ3, (Nq..., nelem))
    x3ξ3 = reshape(vgeo.x3ξ3, (Nq..., nelem))

    x1ξ1 .= x1ξ2 .= x1ξ3 .= zero(T)
    x2ξ1 .= x2ξ2 .= x2ξ3 .= zero(T)
    x3ξ1 .= x3ξ2 .= x3ξ3 .= zero(T)

    @inbounds for e in 1:nelem
        for k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]
            for n in 1:Nq[1]
                x1ξ1[i, j, k, e] += D1[i, n] * x1[n, j, k, e]
                x2ξ1[i, j, k, e] += D1[i, n] * x2[n, j, k, e]
                x3ξ1[i, j, k, e] += D1[i, n] * x3[n, j, k, e]
            end
            for n in 1:Nq[2]
                x1ξ2[i, j, k, e] += D2[j, n] * x1[i, n, k, e]
                x2ξ2[i, j, k, e] += D2[j, n] * x2[i, n, k, e]
                x3ξ2[i, j, k, e] += D2[j, n] * x3[i, n, k, e]
            end
            for n in 1:Nq[3]
                x1ξ3[i, j, k, e] += D3[k, n] * x1[i, j, n, e]
                x2ξ3[i, j, k, e] += D3[k, n] * x2[i, j, n, e]
                x3ξ3[i, j, k, e] += D3[k, n] * x3[i, j, n, e]
            end
        end
    end

    return vgeo
end

"""
    computemetric!(vgeo, sgeo, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- sgeo::SurfaceGeometry, a struct containing the surface geometric factors
- D::NTuple{1,Int}, tuple with 1-D derivative operator on the device

Compute the 1-D metric terms from the element grid arrays `vgeo.x1`. All the arrays
are preallocated by the user and the (square) derivative matrix `D` should be
consistent with the reference grid `ξ1` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x1), Nq)` then the volume arrays
`x1`, `J`, and `ξ1x1` should all have length `Nq * nelem`.  Similarly, the face
arrays `sJ` and `n1` should be of length `nface * nelem` with `nface = 2`.
"""
function computemetric!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    sgeo::SurfaceGeometry{Nfp, <:AbstractArray},
    D::NTuple{1, Matrix{FT}},
) where {Nq, Nfp, FT}

    @assert Nq == map(d -> size(d, 1), D)

    nelem = div(length(vgeo.ωJ), Nq[1])
    ωJ = reshape(vgeo.ωJ, (Nq[1], nelem))
    nface = 2
    n1 = reshape(sgeo.n1, (1, nface, nelem))
    sωJ = reshape(sgeo.sωJ, (1, nface, nelem))

    # Compute vertical Jacobian determinant, JcV, and Jacobian determinant, det(∂x/∂ξ), per quadrature point
    vgeo.JcV .= vgeo.x1ξ1
    vgeo.ωJ .= vgeo.x1ξ1

    vgeo.ξ1x1 .= 1 ./ vgeo.ωJ

    sgeo.n1[1, 1, :] .= -sign.(ωJ[1, :])
    sgeo.n1[1, 2, :] .= sign.(ωJ[Nq[1], :])
    sgeo.sωJ .= 1
    nothing
end

"""
    computemetric!(vgeo, sgeo, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- sgeo::SurfaceGeometry, a struct containing the surface geometric factors
- D::NTuple{2,Int}, a tuple of derivative matrices, i.e., D = (D1, D2), where:
    - D1::DAT2, 1-D derivative operator on the device in the first dimension
    - D2::DAT2, 1-D derivative operator on the device in the second dimension

Compute the 2-D metric terms from the element grid arrays `vgeo.x1` and `vgeo.x2`. All the
arrays are preallocated by the user and the (square) derivative matrice `D1` and
`D2` should be consistent with the reference grid `ξ1` and `ξ2` used in
[`creategrid!`](@ref).

If `Nq = (size(D1, 1), size(D2, 1))` and `nelem = div(length(vgeo.x1), prod(Nq))`
then the volume arrays `vgeo.x1`, `vgeo.x2`, `vgeo.ωJ`, `vgeo.ξ1x1`, `vgeo.ξ2x1`, `vgeo.ξ1x2`, and `vgeo.ξ2x2`
should all be of size `(Nq..., nelem)`.  Similarly, the face arrays `sgeo.sωJ`, `sgeo.n1`,
and `sgeo.n2` should be of size `(maximum(Nq), nface, nelem)` with `nface = 4`
"""
function computemetric!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    sgeo::SurfaceGeometry{Nfp, <:AbstractArray},
    D::NTuple{2, Matrix{FT}},
) where {Nq, Nfp, FT}

    @assert Nq == map(d -> size(d, 1), D)
    @assert Nfp == div.(prod(Nq), Nq)

    nelem = div(length(vgeo.ωJ), prod(Nq))
    x1 = reshape(vgeo.x1, (Nq..., nelem))
    x2 = reshape(vgeo.x2, (Nq..., nelem))
    ωJ = reshape(vgeo.ωJ, (Nq..., nelem))
    JcV = reshape(vgeo.JcV, (Nq..., nelem))
    ξ1x1 = reshape(vgeo.ξ1x1, (Nq..., nelem))
    ξ2x1 = reshape(vgeo.ξ2x1, (Nq..., nelem))
    ξ1x2 = reshape(vgeo.ξ1x2, (Nq..., nelem))
    ξ2x2 = reshape(vgeo.ξ2x2, (Nq..., nelem))
    x1ξ1 = reshape(vgeo.x1ξ1, (Nq..., nelem))
    x1ξ2 = reshape(vgeo.x1ξ2, (Nq..., nelem))
    x2ξ1 = reshape(vgeo.x2ξ1, (Nq..., nelem))
    x2ξ2 = reshape(vgeo.x2ξ2, (Nq..., nelem))
    nface = 4
    n1 = reshape(sgeo.n1, (maximum(Nfp), nface, nelem))
    n2 = reshape(sgeo.n2, (maximum(Nfp), nface, nelem))
    sωJ = reshape(sgeo.sωJ, (maximum(Nfp), nface, nelem))

    for e in 1:nelem
        for j in 1:Nq[2], i in 1:Nq[1]

            # Compute vertical Jacobian determinant, JcV, per quadrature point
            JcV[i, j, e] = hypot(x1ξ2[i, j, e], x2ξ2[i, j, e])
            # Compute Jacobian determinant, det(∂x/∂ξ), per quadrature point
            ωJ[i, j, e] =
                x1ξ1[i, j, e] * x2ξ2[i, j, e] - x2ξ1[i, j, e] * x1ξ2[i, j, e]

            ξ1x1[i, j, e] = x2ξ2[i, j, e] / ωJ[i, j, e]
            ξ2x1[i, j, e] = -x2ξ1[i, j, e] / ωJ[i, j, e]
            ξ1x2[i, j, e] = -x1ξ2[i, j, e] / ωJ[i, j, e]
            ξ2x2[i, j, e] = x1ξ1[i, j, e] / ωJ[i, j, e]
        end

        # Compute surface struct field entries
        for i in 1:maximum(Nfp)
            if i <= Nfp[1]
                sgeo.n1[i, 1, e] = -ωJ[1, i, e] * ξ1x1[1, i, e]
                sgeo.n2[i, 1, e] = -ωJ[1, i, e] * ξ1x2[1, i, e]
                sgeo.n1[i, 2, e] = ωJ[Nq[1], i, e] * ξ1x1[Nq[1], i, e]
                sgeo.n2[i, 2, e] = ωJ[Nq[1], i, e] * ξ1x2[Nq[1], i, e]
            else
                sgeo.n1[i, 1:2, e] .= NaN
                sgeo.n2[i, 1:2, e] .= NaN
            end
            if i <= Nfp[2]
                sgeo.n1[i, 3, e] = -ωJ[i, 1, e] * ξ2x1[i, 1, e]
                sgeo.n2[i, 3, e] = -ωJ[i, 1, e] * ξ2x2[i, 1, e]
                sgeo.n1[i, 4, e] = ωJ[i, Nq[2], e] * ξ2x1[i, Nq[2], e]
                sgeo.n2[i, 4, e] = ωJ[i, Nq[2], e] * ξ2x2[i, Nq[2], e]
            else
                sgeo.n1[i, 3:4, e] .= NaN
                sgeo.n2[i, 3:4, e] .= NaN
            end

            for n in 1:nface
                sgeo.sωJ[i, n, e] = hypot(n1[i, n, e], n2[i, n, e])
                sgeo.n1[i, n, e] /= sωJ[i, n, e]
                sgeo.n2[i, n, e] /= sωJ[i, n, e]
            end
        end

    end

    nothing
end

"""
    computemetric!(vgeo, sgeo, D)

Input arguments:
- vgeo::VolumeGeometry, a struct containing the volumetric geometric factors
- sgeo::SurfaceGeometry, a struct containing the surface geometric factors
- D::NTuple{3,Int}, a tuple of derivative matrices, i.e., D = (D1, D2, D3), where:
    - D1::DAT2, 1-D derivative operator on the device in the first dimension
    - D2::DAT2, 1-D derivative operator on the device in the second dimension
    - D3::DAT2, 1-D derivative operator on the device in the third dimension

Compute the 3-D metric terms from the element grid arrays `vgeo.x1`, `vgeo.x2`, and `vgeo.x3`.
All the arrays are preallocated by the user and the (square) derivative matrice `D1`,
`D2`, and `D3` should be consistent with the reference grid `ξ1`, `ξ2`, and `ξ3` used in
[`creategrid!`](@ref).

If `Nq = size(D1, 1)` and `nelem = div(length(vgeo.x1), Nq^3)` then the volume arrays
`vgeo.x1`, `vgeo.x2`, `vgeo.x3`, `vgeo.ωJ`, `vgeo.ξ1x1`, `vgeo.ξ2x1`, `vgeo.ξ3x1`,
`vgeo.ξ1x2`, `vgeo.ξ2x2`, `vgeo.ξ3x2`, `vgeo.ξ1x3`,`vgeo.ξ2x3`, and `vgeo.ξ3x3`
should all be of length `Nq^3 * nelem`.  Similarly, the face
arrays `sgeo.sωJ`, `sgeo.n1`, `sgeo.n2`, and `sgeo.n3` should be of size `Nq^2 * nface * nelem` with
`nface = 6`.

The curl invariant formulation of Kopriva (2006), equation 37, is used.

Reference:
 - [Kopriva2006](@cite)
"""
function computemetric!(
    vgeo::VolumeGeometry{Nq, <:AbstractArray, <:AbstractArray},
    sgeo::SurfaceGeometry{Nfp, <:AbstractArray},
    D::NTuple{3, Matrix{FT}},
) where {Nq, Nfp, FT}

    @assert Nq == map(d -> size(d, 1), D)
    @assert Nfp == div.(prod(Nq), Nq)

    T = eltype(vgeo.x1)
    nelem = div(length(vgeo.ωJ), prod(Nq))
    x1 = reshape(vgeo.x1, (Nq..., nelem))
    x2 = reshape(vgeo.x2, (Nq..., nelem))
    x3 = reshape(vgeo.x3, (Nq..., nelem))
    ωJ = reshape(vgeo.ωJ, (Nq..., nelem))
    JcV = reshape(vgeo.JcV, (Nq..., nelem))
    ξ1x1 = reshape(vgeo.ξ1x1, (Nq..., nelem))
    ξ2x1 = reshape(vgeo.ξ2x1, (Nq..., nelem))
    ξ3x1 = reshape(vgeo.ξ3x1, (Nq..., nelem))
    ξ1x2 = reshape(vgeo.ξ1x2, (Nq..., nelem))
    ξ2x2 = reshape(vgeo.ξ2x2, (Nq..., nelem))
    ξ3x2 = reshape(vgeo.ξ3x2, (Nq..., nelem))
    ξ1x3 = reshape(vgeo.ξ1x3, (Nq..., nelem))
    ξ2x3 = reshape(vgeo.ξ2x3, (Nq..., nelem))
    ξ3x3 = reshape(vgeo.ξ3x3, (Nq..., nelem))
    x1ξ1 = reshape(vgeo.x1ξ1, (Nq..., nelem))
    x1ξ2 = reshape(vgeo.x1ξ2, (Nq..., nelem))
    x1ξ3 = reshape(vgeo.x1ξ3, (Nq..., nelem))
    x2ξ1 = reshape(vgeo.x2ξ1, (Nq..., nelem))
    x2ξ2 = reshape(vgeo.x2ξ2, (Nq..., nelem))
    x2ξ3 = reshape(vgeo.x2ξ3, (Nq..., nelem))
    x3ξ1 = reshape(vgeo.x3ξ1, (Nq..., nelem))
    x3ξ2 = reshape(vgeo.x3ξ2, (Nq..., nelem))
    x3ξ3 = reshape(vgeo.x3ξ3, (Nq..., nelem))

    nface = 6
    n1 = reshape(sgeo.n1, maximum(Nfp), nface, nelem)
    n2 = reshape(sgeo.n2, maximum(Nfp), nface, nelem)
    n3 = reshape(sgeo.n3, maximum(Nfp), nface, nelem)
    sωJ = reshape(sgeo.sωJ, maximum(Nfp), nface, nelem)

    JI2 = similar(vgeo.ωJ, Nq...)
    (yzr, yzs, yzt) = (similar(JI2), similar(JI2), similar(JI2))
    (zxr, zxs, zxt) = (similar(JI2), similar(JI2), similar(JI2))
    (xyr, xys, xyt) = (similar(JI2), similar(JI2), similar(JI2))
    # Temporary variables to compute inverse of a 3x3 matrix
    (a11, a12, a13) = (similar(JI2), similar(JI2), similar(JI2))
    (a21, a22, a23) = (similar(JI2), similar(JI2), similar(JI2))
    (a31, a32, a33) = (similar(JI2), similar(JI2), similar(JI2))

    ξ1x1 .= ξ2x1 .= ξ3x1 .= zero(T)
    ξ1x2 .= ξ2x2 .= ξ3x2 .= zero(T)
    ξ1x3 .= ξ2x3 .= ξ3x3 .= zero(T)

    fill!(n1, NaN)
    fill!(n2, NaN)
    fill!(n3, NaN)
    fill!(sωJ, NaN)

    @inbounds for e in 1:nelem
        for k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]

            # Compute vertical Jacobian determinant, JcV, per quadrature point
            JcV[i, j, k, e] =
                hypot(x1ξ3[i, j, k, e], x2ξ3[i, j, k, e], x3ξ3[i, j, k, e])
            # Compute Jacobian determinant, det(∂x/∂ξ), per quadrature point
            ωJ[i, j, k, e] = (
                x1ξ1[i, j, k, e] * (
                    x2ξ2[i, j, k, e] * x3ξ3[i, j, k, e] -
                    x3ξ2[i, j, k, e] * x2ξ3[i, j, k, e]
                ) +
                x2ξ1[i, j, k, e] * (
                    x3ξ2[i, j, k, e] * x1ξ3[i, j, k, e] -
                    x1ξ2[i, j, k, e] * x3ξ3[i, j, k, e]
                ) +
                x3ξ1[i, j, k, e] * (
                    x1ξ2[i, j, k, e] * x2ξ3[i, j, k, e] -
                    x2ξ2[i, j, k, e] * x1ξ3[i, j, k, e]
                )
            )

            JI2[i, j, k] = 1 / (2 * ωJ[i, j, k, e])

            yzr[i, j, k] =
                x2[i, j, k, e] * x3ξ1[i, j, k, e] -
                x3[i, j, k, e] * x2ξ1[i, j, k, e]
            yzs[i, j, k] =
                x2[i, j, k, e] * x3ξ2[i, j, k, e] -
                x3[i, j, k, e] * x2ξ2[i, j, k, e]
            yzt[i, j, k] =
                x2[i, j, k, e] * x3ξ3[i, j, k, e] -
                x3[i, j, k, e] * x2ξ3[i, j, k, e]
            zxr[i, j, k] =
                x3[i, j, k, e] * x1ξ1[i, j, k, e] -
                x1[i, j, k, e] * x3ξ1[i, j, k, e]
            zxs[i, j, k] =
                x3[i, j, k, e] * x1ξ2[i, j, k, e] -
                x1[i, j, k, e] * x3ξ2[i, j, k, e]
            zxt[i, j, k] =
                x3[i, j, k, e] * x1ξ3[i, j, k, e] -
                x1[i, j, k, e] * x3ξ3[i, j, k, e]
            xyr[i, j, k] =
                x1[i, j, k, e] * x2ξ1[i, j, k, e] -
                x2[i, j, k, e] * x1ξ1[i, j, k, e]
            xys[i, j, k] =
                x1[i, j, k, e] * x2ξ2[i, j, k, e] -
                x2[i, j, k, e] * x1ξ2[i, j, k, e]
            xyt[i, j, k] =
                x1[i, j, k, e] * x2ξ3[i, j, k, e] -
                x2[i, j, k, e] * x1ξ3[i, j, k, e]
        end

        for k in 1:Nq[3], j in 1:Nq[2], i in 1:Nq[1]
            for n in 1:Nq[1]
                ξ2x1[i, j, k, e] -= D[1][i, n] * yzt[n, j, k]
                ξ3x1[i, j, k, e] += D[1][i, n] * yzs[n, j, k]
                ξ2x2[i, j, k, e] -= D[1][i, n] * zxt[n, j, k]
                ξ3x2[i, j, k, e] += D[1][i, n] * zxs[n, j, k]
                ξ2x3[i, j, k, e] -= D[1][i, n] * xyt[n, j, k]
                ξ3x3[i, j, k, e] += D[1][i, n] * xys[n, j, k]
            end
            for n in 1:Nq[2]
                ξ1x1[i, j, k, e] += D[2][j, n] * yzt[i, n, k]
                ξ3x1[i, j, k, e] -= D[2][j, n] * yzr[i, n, k]
                ξ1x2[i, j, k, e] += D[2][j, n] * zxt[i, n, k]
                ξ3x2[i, j, k, e] -= D[2][j, n] * zxr[i, n, k]
                ξ1x3[i, j, k, e] += D[2][j, n] * xyt[i, n, k]
                ξ3x3[i, j, k, e] -= D[2][j, n] * xyr[i, n, k]
            end
            for n in 1:Nq[3]
                ξ1x1[i, j, k, e] -= D[3][k, n] * yzs[i, j, n]
                ξ2x1[i, j, k, e] += D[3][k, n] * yzr[i, j, n]
                ξ1x2[i, j, k, e] -= D[3][k, n] * zxs[i, j, n]
                ξ2x2[i, j, k, e] += D[3][k, n] * zxr[i, j, n]
                ξ1x3[i, j, k, e] -= D[3][k, n] * xys[i, j, n]
                ξ2x3[i, j, k, e] += D[3][k, n] * xyr[i, j, n]
            end
            ξ1x1[i, j, k, e] *= JI2[i, j, k]
            ξ2x1[i, j, k, e] *= JI2[i, j, k]
            ξ3x1[i, j, k, e] *= JI2[i, j, k]
            ξ1x2[i, j, k, e] *= JI2[i, j, k]
            ξ2x2[i, j, k, e] *= JI2[i, j, k]
            ξ3x2[i, j, k, e] *= JI2[i, j, k]
            ξ1x3[i, j, k, e] *= JI2[i, j, k]
            ξ2x3[i, j, k, e] *= JI2[i, j, k]
            ξ3x3[i, j, k, e] *= JI2[i, j, k]


            # Invert ∂ξk/∂xi, since the discrete curl-invariant form that we have
            # just computed, ∂ξk/∂xi, is not equal to its inverse
            a11[i, j, k] =
                ξ2x2[i, j, k, e] * ξ3x3[i, j, k, e] -
                ξ2x3[i, j, k, e] * ξ3x2[i, j, k, e]
            a12[i, j, k] =
                ξ1x3[i, j, k, e] * ξ3x2[i, j, k, e] -
                ξ1x2[i, j, k, e] * ξ3x3[i, j, k, e]
            a13[i, j, k] =
                ξ1x2[i, j, k, e] * ξ2x3[i, j, k, e] -
                ξ1x3[i, j, k, e] * ξ2x2[i, j, k, e]
            a21[i, j, k] =
                ξ2x3[i, j, k, e] * ξ3x1[i, j, k, e] -
                ξ2x1[i, j, k, e] * ξ3x3[i, j, k, e]
            a22[i, j, k] =
                ξ1x1[i, j, k, e] * ξ3x3[i, j, k, e] -
                ξ1x3[i, j, k, e] * ξ3x1[i, j, k, e]
            a23[i, j, k] =
                ξ1x3[i, j, k, e] * ξ2x1[i, j, k, e] -
                ξ1x1[i, j, k, e] * ξ2x3[i, j, k, e]
            a31[i, j, k] =
                ξ2x1[i, j, k, e] * ξ3x2[i, j, k, e] -
                ξ2x2[i, j, k, e] * ξ3x1[i, j, k, e]
            a32[i, j, k] =
                ξ1x2[i, j, k, e] * ξ3x1[i, j, k, e] -
                ξ1x1[i, j, k, e] * ξ3x2[i, j, k, e]
            a33[i, j, k] =
                ξ1x1[i, j, k, e] * ξ2x2[i, j, k, e] -
                ξ1x2[i, j, k, e] * ξ2x1[i, j, k, e]

            det =
                ξ1x1[i, j, k, e] * a11[i, j, k] +
                ξ2x1[i, j, k, e] * a12[i, j, k] +
                ξ3x1[i, j, k, e] * a13[i, j, k]

            x1ξ1[i, j, k, e] =
                1.0 / det * (
                    a11[i, j, k] * a11[i, j, k] +
                    a12[i, j, k] * a12[i, j, k] +
                    a13[i, j, k] * a13[i, j, k]
                )
            x1ξ2[i, j, k, e] =
                1.0 / det * (
                    a11[i, j, k] * a21[i, j, k] +
                    a12[i, j, k] * a22[i, j, k] +
                    a13[i, j, k] * a23[i, j, k]
                )
            x1ξ3[i, j, k, e] =
                1.0 / det * (
                    a11[i, j, k] * a31[i, j, k] +
                    a12[i, j, k] * a32[i, j, k] +
                    a13[i, j, k] * a33[i, j, k]
                )
            x2ξ1[i, j, k, e] =
                1.0 / det * (
                    a21[i, j, k] * a11[i, j, k] +
                    a22[i, j, k] * a12[i, j, k] +
                    a23[i, j, k] * a13[i, j, k]
                )
            x2ξ2[i, j, k, e] =
                1.0 / det * (
                    a21[i, j, k] * a21[i, j, k] +
                    a22[i, j, k] * a22[i, j, k] +
                    a23[i, j, k] * a23[i, j, k]
                )
            x2ξ3[i, j, k, e] =
                1.0 / det * (
                    a21[i, j, k] * a31[i, j, k] +
                    a22[i, j, k] * a32[i, j, k] +
                    a23[i, j, k] * a33[i, j, k]
                )
            x3ξ1[i, j, k, e] =
                1.0 / det * (
                    a31[i, j, k] * a11[i, j, k] +
                    a32[i, j, k] * a12[i, j, k] +
                    a33[i, j, k] * a13[i, j, k]
                )
            x3ξ2[i, j, k, e] =
                1.0 / det * (
                    a31[i, j, k] * a21[i, j, k] +
                    a32[i, j, k] * a22[i, j, k] +
                    a33[i, j, k] * a23[i, j, k]
                )
            x3ξ3[i, j, k, e] =
                1.0 / det * (
                    a31[i, j, k] * a31[i, j, k] +
                    a32[i, j, k] * a32[i, j, k] +
                    a33[i, j, k] * a33[i, j, k]
                )
        end

        # Compute surface struct field entries
        # faces 1 & 2
        for k in 1:Nq[3], j in 1:Nq[2]
            n = j + (k - 1) * Nq[2]
            sgeo.n1[n, 1, e] = -ωJ[1, j, k, e] * ξ1x1[1, j, k, e]
            sgeo.n2[n, 1, e] = -ωJ[1, j, k, e] * ξ1x2[1, j, k, e]
            sgeo.n3[n, 1, e] = -ωJ[1, j, k, e] * ξ1x3[1, j, k, e]
            sgeo.n1[n, 2, e] = ωJ[Nq[1], j, k, e] * ξ1x1[Nq[1], j, k, e]
            sgeo.n2[n, 2, e] = ωJ[Nq[1], j, k, e] * ξ1x2[Nq[1], j, k, e]
            sgeo.n3[n, 2, e] = ωJ[Nq[1], j, k, e] * ξ1x3[Nq[1], j, k, e]
            for f in 1:2
                sgeo.sωJ[n, f, e] = hypot(n1[n, f, e], n2[n, f, e], n3[n, f, e])
                sgeo.n1[n, f, e] /= sωJ[n, f, e]
                sgeo.n2[n, f, e] /= sωJ[n, f, e]
                sgeo.n3[n, f, e] /= sωJ[n, f, e]
            end
        end
        # faces 3 & 4
        for k in 1:Nq[3], i in 1:Nq[1]
            n = i + (k - 1) * Nq[1]
            sgeo.n1[n, 3, e] = -ωJ[i, 1, k, e] * ξ2x1[i, 1, k, e]
            sgeo.n2[n, 3, e] = -ωJ[i, 1, k, e] * ξ2x2[i, 1, k, e]
            sgeo.n3[n, 3, e] = -ωJ[i, 1, k, e] * ξ2x3[i, 1, k, e]
            sgeo.n1[n, 4, e] = ωJ[i, Nq[2], k, e] * ξ2x1[i, Nq[2], k, e]
            sgeo.n2[n, 4, e] = ωJ[i, Nq[2], k, e] * ξ2x2[i, Nq[2], k, e]
            sgeo.n3[n, 4, e] = ωJ[i, Nq[2], k, e] * ξ2x3[i, Nq[2], k, e]
            for f in 3:4
                sgeo.sωJ[n, f, e] = hypot(n1[n, f, e], n2[n, f, e], n3[n, f, e])
                sgeo.n1[n, f, e] /= sωJ[n, f, e]
                sgeo.n2[n, f, e] /= sωJ[n, f, e]
                sgeo.n3[n, f, e] /= sωJ[n, f, e]
            end
        end
        # faces 5 & 6
        for j in 1:Nq[2], i in 1:Nq[1]
            n = i + (j - 1) * Nq[1]
            sgeo.n1[n, 5, e] = -ωJ[i, j, 1, e] * ξ3x1[i, j, 1, e]
            sgeo.n2[n, 5, e] = -ωJ[i, j, 1, e] * ξ3x2[i, j, 1, e]
            sgeo.n3[n, 5, e] = -ωJ[i, j, 1, e] * ξ3x3[i, j, 1, e]
            sgeo.n1[n, 6, e] = ωJ[i, j, Nq[3], e] * ξ3x1[i, j, Nq[3], e]
            sgeo.n2[n, 6, e] = ωJ[i, j, Nq[3], e] * ξ3x2[i, j, Nq[3], e]
            sgeo.n3[n, 6, e] = ωJ[i, j, Nq[3], e] * ξ3x3[i, j, Nq[3], e]
            for f in 5:6
                sgeo.sωJ[n, f, e] = hypot(n1[n, f, e], n2[n, f, e], n3[n, f, e])
                sgeo.n1[n, f, e] /= sωJ[n, f, e]
                sgeo.n2[n, f, e] /= sωJ[n, f, e]
                sgeo.n3[n, f, e] /= sωJ[n, f, e]
            end
        end
    end

    nothing
end

end # module
