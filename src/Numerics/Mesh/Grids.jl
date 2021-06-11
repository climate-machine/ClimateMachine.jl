module Grids
using ..Topologies, ..GeometricFactors
import ..Metrics, ..Elements
import ..BrickMesh
using ClimateMachine.MPIStateArrays

using MPI
using LinearAlgebra
using KernelAbstractions
using StaticArrays
using DocStringExtensions

export DiscontinuousSpectralElementGrid, QuadratureGrid, AbstractGrid
export dofs_per_element, arraytype, dimensionality, polynomialorders
export referencepoints, min_node_distance, get_z, computegeometry
export EveryDirection, HorizontalDirection, VerticalDirection, Direction
export tpxv!

abstract type Direction end
struct EveryDirection <: Direction end
struct HorizontalDirection <: Direction end
struct VerticalDirection <: Direction end
Base.in(::T, ::S) where {T <: Direction, S <: Direction} = T == S

"""
    MinNodalDistance{FT}

A struct containing the minimum nodal distance
along horizontal and vertical directions.
"""
struct MinNodalDistance{FT}
    "horizontal"
    h::FT
    "vertical"
    v::FT
end

abstract type AbstractGrid{
    FloatType,
    dim,
    polynomialorder,
    numberofDOFs,
    DeviceArray,
} end

include("TensorProduct.jl")
include("FastTensorProduct.jl")

dofs_per_element(::AbstractGrid{T, D, N, Np}) where {T, D, N, Np} = Np

polynomialorders(::AbstractGrid{T, dim, N}) where {T, dim, N} = N

dimensionality(::AbstractGrid{T, dim}) where {T, dim} = dim

Base.eltype(::AbstractGrid{T}) where {T} = T

arraytype(::AbstractGrid{T, D, N, Np, DA}) where {T, D, N, Np, DA} = DA

"""
    referencepoints(::AbstractGrid)

Returns the points on the reference element.
"""
referencepoints(::AbstractGrid) = error("needs to be implemented")

"""
    min_node_distance(::AbstractGrid, direction::Direction=EveryDirection() )

Returns an approximation of the minimum node distance in physical space.
"""
function min_node_distance(
    ::AbstractGrid,
    direction::Direction = EveryDirection(),
)
    error("needs to be implemented")
end

# {{{
# `vgeo` stores geometry and metrics terms at the volume quadrature /
# interpolation points
const _nvgeo = 16
const _ξ1x1,
_ξ2x1,
_ξ3x1,
_ξ1x2,
_ξ2x2,
_ξ3x2,
_ξ1x3,
_ξ2x3,
_ξ3x3,
_M,
_MI,
_MH,
_x1,
_x2,
_x3,
_JcV = 1:_nvgeo
const vgeoid = (
    # ∂ξk/∂xi: derivative of the Cartesian reference element coordinate `ξ_k`
    # with respect to the Cartesian physical coordinate `x_i`
    ξ1x1id = _ξ1x1,
    ξ2x1id = _ξ2x1,
    ξ3x1id = _ξ3x1,
    ξ1x2id = _ξ1x2,
    ξ2x2id = _ξ2x2,
    ξ3x2id = _ξ3x2,
    ξ1x3id = _ξ1x3,
    ξ2x3id = _ξ2x3,
    ξ3x3id = _ξ3x3,
    # M refers to the mass matrix. This is the physical mass matrix, and thus
    # contains the Jacobian determinant:
    #    J .* (ωᵢ ⊗ ωⱼ ⊗ ωₖ)
    # where ωᵢ are the quadrature weights and J is the Jacobian determinant
    # det(∂x/∂ξ)
    Mid = _M,
    # Inverse mass matrix: 1 ./ M
    MIid = _MI,
    # Horizontal mass matrix (used in diagnostics)
    #    J .* norm(∂ξ3/∂x) * (ωᵢ ⊗ ωⱼ); for integrating over a plane
    # (in 2-D ξ2 not ξ3 is used)
    MHid = _MH,
    # Nodal degrees of freedom locations in Cartesian physical space
    x1id = _x1,
    x2id = _x2,
    x3id = _x3,
    # Metric terms for vertical line integrals
    #   norm(∂x/∂ξ3)
    # (in 2-D ξ2 not ξ3 is used)
    JcVid = _JcV,
)

# `sgeo` stores geometry and metrics terms at the surface quadrature /
# interpolation points
const _nsgeo = 5
const _n1, _n2, _n3, _sM, _vMI = 1:_nsgeo
const sgeoid = (
    # outward pointing unit normal in physical space
    n1id = _n1,
    n2id = _n2,
    n3id = _n3,
    # sM refers to the surface mass matrix. This is the physical mass matrix,
    # and thus contains the surface Jacobian determinant:
    #    sJ .* (ωⱼ ⊗ ωₖ)
    # where ωᵢ are the quadrature weights and sJ is the surface Jacobian
    # determinant
    sMid = _sM,
    # Volume mass matrix at the surface nodes (needed in the lift operation,
    # i.e., the projection of a face field back to the volume). Since DGSEM is
    # used only collocated, volume mass matrices are required.
    vMIid = _vMI,
)
# }}}

"""
    DiscontinuousSpectralElementGrid(topology; FloatType, DeviceArray,
                                     polynomialorder,
                                     meshwarp = (x...)->identity(x))

Generate a discontinuous spectral element (tensor product,
Legendre-Gauss-Lobatto) grid/mesh from a `topology`, where the order of the
elements is given by `polynomialorder`. `DeviceArray` gives the array type used
to store the data (`CuArray` or `Array`), and the coordinate points will be of
`FloatType`.

The polynomial order can be different in each direction (specified as a
`NTuple`). If only a single integer is specified, then each dimension will use
the same order. If the topology dimension is 3 and the `polynomialorder` has
dimension 2, then the first value will be used for horizontal and the second for
the vertical.

The optional `meshwarp` function allows the coordinate points to be warped after
the mesh is created; the mesh degrees of freedom are orginally assigned using a
trilinear blend of the element corner locations.
"""
struct DiscontinuousSpectralElementGrid{
    T,
    dim,
    N,
    Np,
    DA,
    DAT1,
    DAT2,
    DAT3,
    DAT4,
    DAI1,
    DAI2,
    DAI3,
    TOP,
    TVTK,
    MINΔ,
} <: AbstractGrid{T, dim, N, Np, DA}
    "mesh topology"
    topology::TOP

    "volume metric terms"
    vgeo::DAT3

    "surface metric terms"
    sgeo::DAT4

    "element to boundary condition map"
    elemtobndy::DAI2

    "volume DOF to element minus side map"
    vmap⁻::DAI3

    "volume DOF to element plus side map"
    vmap⁺::DAI3

    "list of DOFs that need to be received (in neighbors order)"
    vmaprecv::DAI1

    "list of DOFs that need to be sent (in neighbors order)"
    vmapsend::DAI1

    "An array of ranges in `vmaprecv` to receive from each neighbor"
    nabrtovmaprecv::Any

    "An array of ranges in `vmapsend` to send to each neighbor"
    nabrtovmapsend::Any

    "Array of real elements that do not have a ghost element as a neighbor"
    interiorelems::Any

    "Array of real elements that have at least one ghost element as a neighbor"
    exteriorelems::Any

    "Array indicating if a degree of freedom (real or ghost) is active"
    activedofs::Any

    "1-D LGL weights on the device (one for each dimension)"
    ω::DAT1

    "1-D derivative operator on the device (one for each dimension)"
    D::DAT2

    "Transpose of 1-D derivative operator on the device (one for each dimension)"
    Dᵀ::DAT2

    "1-D indefinite integral operator on the device (one for each dimension)"
    Imat::DAT2

    """
    tuple of (x1, x2, x3) to use for vtk output (Needed for the `N = 0` case) in
    other cases these match `vgeo` values
    """
    x_vtk::TVTK

    """
    Minimum nodal distances for horizontal and vertical directions
    """
    minΔ::MINΔ

    """
    Map to vertex degrees of freedom: `vertmap[v]` contains the degree of freedom located at vertex `v`.
    """
    vertmap::Union{DAI1, Nothing}

    """
    Map to edge degrees of freedom: `edgemap[i, edgno, orient]` contains the element node index of 
    the `i`th interior node on edge `edgno`, under orientation `orient`.
    """
    edgemap::Union{DAI3, Nothing}

    """
    Map to face degrees of freedom: `facemap[ij, fcno, orient]` contains the element node index of the `ij`th 
    interior node on face `fcno` under orientation `orient`.

    Note that only the two orientations that are generated for stacked meshes are currently supported, i.e.,
    mesh orientation `3` as defined by `BrickMesh` gets mapped to orientation `2` for this data structure.    
    """
    facemap::Union{DAI3, Nothing}

    "Temporary Storage on the spectral element mesh"
    scratch::DAT3

    "Temporary Storage for fast tensor product"
    scratch_ftp::DAT3
    # Constructor for a tuple of polynomial orders
    function DiscontinuousSpectralElementGrid(
        topology::AbstractTopology{dim};
        polynomialorder,
        FloatType,
        DeviceArray,
        meshwarp::Function = (x...) -> identity(x),
    ) where {dim}

        if polynomialorder isa Integer
            polynomialorder = ntuple(j -> polynomialorder, dim)
        elseif polynomialorder isa NTuple{2} && dim == 3
            polynomialorder =
                (polynomialorder[1], polynomialorder[1], polynomialorder[2])
        end

        @assert dim == length(polynomialorder)

        N = polynomialorder

        (vmap⁻, vmap⁺) = mappings(
            N,
            topology.elemtoelem,
            topology.elemtoface,
            topology.elemtoordr,
        )

        (vmaprecv, nabrtovmaprecv) = commmapping(
            N,
            topology.ghostelems,
            topology.ghostfaces,
            topology.nabrtorecv,
        )
        (vmapsend, nabrtovmapsend) = commmapping(
            N,
            topology.sendelems,
            topology.sendfaces,
            topology.nabrtosend,
        )

        Np = prod(N .+ 1)

        vertmap, edgemap, facemap = init_vertex_edge_face_mappings(N)
        # Create element operators for each polynomial order
        ξω = ntuple(
            j ->
                N[j] == 0 ? Elements.glpoints(FloatType, N[j]) :
                Elements.lglpoints(FloatType, N[j]),
            dim,
        )
        ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)

        Imat = ntuple(
            j -> indefinite_integral_interpolation_matrix(ξ[j], ω[j]),
            dim,
        )
        D = ntuple(j -> Elements.spectralderivative(ξ[j]), dim)
        Dᵀ = ntuple(j -> Array(transpose(D[j])), dim)

        (vgeo, sgeo, x_vtk) =
            computegeometry(topology.elemtocoord, D, ξ, ω, meshwarp)

        vgeo = vgeo.array
        sgeo = sgeo.array
        @assert Np == size(vgeo, 1)

        activedofs = zeros(Bool, Np * length(topology.elems))
        activedofs[1:(Np * length(topology.realelems))] .= true
        activedofs[vmaprecv] .= true

        # Create arrays on the device
        vgeo = DeviceArray(vgeo)
        sgeo = DeviceArray(sgeo)
        elemtobndy = DeviceArray(topology.elemtobndy)
        vmap⁻ = DeviceArray(vmap⁻)
        vmap⁺ = DeviceArray(vmap⁺)
        vmapsend = DeviceArray(vmapsend)
        vmaprecv = DeviceArray(vmaprecv)
        activedofs = DeviceArray(activedofs)
        ω = DeviceArray.(ω)
        D = DeviceArray.(D)
        Dᵀ = DeviceArray.(Dᵀ)
        Imat = DeviceArray.(Imat)

        # FIXME: There has got to be a better way!
        DAT1 = typeof(ω)
        DAT2 = typeof(D)
        DAT3 = typeof(vgeo)
        DAT4 = typeof(sgeo)
        DAI1 = typeof(vmapsend)
        DAI2 = typeof(elemtobndy)
        DAI3 = typeof(vmap⁻)
        TOP = typeof(topology)
        TVTK = typeof(x_vtk)
        if vertmap isa Array
            vertmap = DAI1(vertmap)
        end
        if edgemap isa Array
            edgemap = DAI3(edgemap)
        end
        if facemap isa Array
            facemap = DAI3(facemap)
        end
        FT = FloatType
        minΔ = MinNodalDistance(
            min_node_distance(vgeo, topology, N, FT, HorizontalDirection()),
            min_node_distance(vgeo, topology, N, FT, VerticalDirection()),
        )
        MINΔ = typeof(minΔ)

        scratch = DeviceArray(Array{FloatType}(
            undef,
            Np,
            9,
            length(topology.realelems),
        ))

        scratch_ftp = DeviceArray(Array{FloatType}(
            undef,
            Np,
            3,
            length(topology.realelems),
        ))

        new{
            FloatType,
            dim,
            N,
            Np,
            DeviceArray,
            DAT1,
            DAT2,
            DAT3,
            DAT4,
            DAI1,
            DAI2,
            DAI3,
            TOP,
            TVTK,
            MINΔ,
        }(
            topology,
            vgeo,
            sgeo,
            elemtobndy,
            vmap⁻,
            vmap⁺,
            vmaprecv,
            vmapsend,
            nabrtovmaprecv,
            nabrtovmapsend,
            DeviceArray(topology.interiorelems),
            DeviceArray(topology.exteriorelems),
            activedofs,
            ω,
            D,
            Dᵀ,
            Imat,
            x_vtk,
            minΔ,
            vertmap,
            edgemap,
            facemap,
            scratch,
            scratch_ftp,
        )
    end
end

"""
    referencepoints(::DiscontinuousSpectralElementGrid)

Returns the 1D interpolation points used for the reference element.
"""
function referencepoints(
    ::DiscontinuousSpectralElementGrid{FT, dim, N},
) where {FT, dim, N}
    ξω = ntuple(
        j ->
            N[j] == 0 ? Elements.glpoints(FT, N[j]) :
            Elements.lglpoints(FT, N[j]),
        dim,
    )
    ξ, _ = ntuple(j -> map(x -> x[j], ξω), 2)
    return ξ
end

"""
    min_node_distance(
        ::DiscontinuousSpectralElementGrid,
        direction::Direction=EveryDirection())
    )

Returns an approximation of the minimum node distance in physical space along
the reference coordinate directions.  The direction controls which reference
directions are considered.
"""
min_node_distance(
    grid::DiscontinuousSpectralElementGrid,
    direction::Direction = EveryDirection(),
) = min_node_distance(grid.minΔ, direction)

min_node_distance(minΔ::MinNodalDistance, ::VerticalDirection) = minΔ.v
min_node_distance(minΔ::MinNodalDistance, ::HorizontalDirection) = minΔ.h
min_node_distance(minΔ::MinNodalDistance, ::EveryDirection) =
    min(minΔ.h, minΔ.v)

function min_node_distance(
    vgeo,
    topology::AbstractTopology{dim},
    N,
    ::Type{T},
    direction::Direction = EveryDirection(),
) where {T, dim}
    topology = topology
    nrealelem = length(topology.realelems)
    if nrealelem > 0
        Nq = N .+ 1
        Np = prod(Nq)
        device = vgeo isa Array ? CPU() : CUDADevice()
        min_neighbor_distance = similar(vgeo, Np, nrealelem)
        event = Event(device)
        event = kernel_min_neighbor_distance!(device, min(Np, 1024))(
            Val(N),
            Val(dim),
            direction,
            min_neighbor_distance,
            vgeo,
            topology.realelems;
            ndrange = (Np * nrealelem),
            dependencies = (event,),
        )
        wait(device, event)
        locmin = minimum(min_neighbor_distance)
    else
        locmin = typemax(T)
    end

    MPI.Allreduce(locmin, min, topology.mpicomm)
end

"""
    get_z(grid; z_scale = 1, rm_dupes = false)

Get the Gauss-Lobatto points along the Z-coordinate.

 - `grid`: DG grid
 - `z_scale`: multiplies `z-coordinate`
 - `rm_dupes`: removes duplicate Gauss-Lobatto points
"""
function get_z(
    grid::DiscontinuousSpectralElementGrid{T, dim, N};
    z_scale = 1,
    rm_dupes = false,
) where {T, dim, N}
    # Assumes same polynomial orders in all horizontal directions
    @assert dim < 3 || N[1] == N[2]
    Nhoriz = N[1]
    Nvert = N[end]
    Nph = (Nhoriz + 1)^2
    Np = Nph * (Nvert + 1)
    if last(polynomialorders(grid)) == 0
        rm_dupes = false # no duplicates in FVM
    end
    if rm_dupes
        ijk_range = (1:Nph:(Np - Nph))
        vgeo = Array(grid.vgeo)
        z = reshape(vgeo[ijk_range, _x3, :], :)
        z = [z..., vgeo[Np, _x3, end]]
        return z * z_scale
    else
        ijk_range = (1:Nph:Np)
        z = Array(reshape(grid.vgeo[ijk_range, _x3, :], :))
        return z * z_scale
    end
    return reshape(grid.vgeo[(1:Nph:Np), _x3, :], :) * z_scale
end

function Base.getproperty(G::DiscontinuousSpectralElementGrid, s::Symbol)
    if s ∈ keys(vgeoid)
        vgeoid[s]
    elseif s ∈ keys(sgeoid)
        sgeoid[s]
    else
        getfield(G, s)
    end
end

function Base.propertynames(G::DiscontinuousSpectralElementGrid)
    (
        fieldnames(DiscontinuousSpectralElementGrid)...,
        keys(vgeoid)...,
        keys(sgeoid)...,
    )
end

# {{{ mappings
"""
    mappings(N, elemtoelem, elemtoface, elemtoordr)

This function takes in a tuple of polynomial orders `N` and parts of a topology
(as returned from `connectmesh`) and returns index mappings for the element
surface flux computation. The returned `Tuple` contains:

 - `vmap⁻` an array of linear indices into the volume degrees of freedom where
   `vmap⁻[:,f,e]` are the degrees of freedom indices for face `f` of element
    `e`.

 - `vmap⁺` an array of linear indices into the volume degrees of freedom where
   `vmap⁺[:,f,e]` are the degrees of freedom indices for the face neighboring
   face `f` of element `e`.
"""
function mappings(N, elemtoelem, elemtoface, elemtoordr)
    nfaces, nelem = size(elemtoelem)

    d = div(nfaces, 2)
    Nq = N .+ 1
    # number of points in the element
    Np = prod(Nq)

    # Compute the maximum number of points on a face
    Nfp = div.(Np, Nq)

    # linear index for each direction, e.g., (i, j, k) -> n
    p = reshape(1:Np, ntuple(j -> Nq[j], d))

    # fmask[f] -> returns an array of all degrees of freedom on face f
    fmask = if d == 1
        (
            p[1:1],    # Face 1
            p[Nq[1]:Nq[1]], # Face 2
        )
    elseif d == 2
        (
            p[1, :][:],     # Face 1
            p[Nq[1], :][:], # Face 2
            p[:, 1][:],     # Face 3
            p[:, Nq[2]][:], # Face 4
        )
    elseif d == 3
        (
            p[1, :, :][:],     # Face 1
            p[Nq[1], :, :][:], # Face 2
            p[:, 1, :][:],     # Face 3
            p[:, Nq[2], :][:], # Face 4
            p[:, :, 1][:],     # Face 5
            p[:, :, Nq[3]][:], # Face 6
        )
    else
        error("unknown dimensionality")
    end

    # Create a map from Cartesian face dof number to linear face dof numbering
    # inds[face][i, j] -> n
    inds = ntuple(
        f -> dropdims(
            LinearIndices(ntuple(j -> j == cld(f, 2) ? 1 : Nq[j], d));
            dims = cld(f, 2),
        ),
        nfaces,
    )

    # Use the largest possible storage
    vmap⁻ = fill!(similar(elemtoelem, maximum(Nfp), nfaces, nelem), 0)
    vmap⁺ = fill!(similar(elemtoelem, maximum(Nfp), nfaces, nelem), 0)

    for e1 in 1:nelem, f1 in 1:nfaces
        e2 = elemtoelem[f1, e1]
        f2 = elemtoface[f1, e1]
        o2 = elemtoordr[f1, e1]
        d1, d2 = cld(f1, 2), cld(f2, 2)

        # Check to make sure the dof grid is conforming
        @assert Nfp[d1] == Nfp[d2]

        # Always pull out minus side without any flips / rotations
        vmap⁻[1:Nfp[d1], f1, e1] .= Np * (e1 - 1) .+ fmask[f1][1:Nfp[d1]][:]

        # Orientation codes defined in BrickMesh.jl (arbitrary numbers in 3D)
        if o2 == 1 # Neighbor oriented same as minus
            vmap⁺[1:Nfp[d1], f1, e1] .= Np * (e2 - 1) .+ fmask[f2][1:Nfp[d1]][:]
        elseif d == 3 && o2 == 3 # Neighbor fliped in first index
            vmap⁺[1:Nfp[d1], f1, e1] =
                Np * (e2 - 1) .+ fmask[f2][inds[f2][end:-1:1, :]][:]
        else
            error("Orientation '$o2' with dim '$d' not supported yet")
        end
    end

    (vmap⁻, vmap⁺)
end
# }}}

function init_vertex_edge_face_mappings(N)
    dim = length(N)
    Np = N .+ 1
    if dim == 3 && Np[end] > 2
        nodes = reshape(1:prod(Np), Np)
        vertmap =
            Int64.([
                nodes[1, 1, 1],
                nodes[Np[1], 1, 1],
                nodes[1, Np[2], 1],
                nodes[Np[1], Np[2], 1],
                nodes[1, 1, Np[3]],
                nodes[Np[1], 1, Np[3]],
                nodes[1, Np[2], Np[3]],
                nodes[Np[1], Np[2], Np[3]],
            ])
        Ne = Np .- 2
        Ne_max = maximum(Ne)
        if Ne_max ≥ 1
            edgemap = -ones(Int64, Ne_max, 12, 2)

            if Np[1] > 2
                edgemap[1:Ne[1], 1, 1] .= nodes[2:(end - 1), 1, 1]
                edgemap[1:Ne[1], 2, 1] .= nodes[2:(end - 1), Np[2], 1]
                edgemap[1:Ne[1], 3, 1] .= nodes[2:(end - 1), 1, Np[3]]
                edgemap[1:Ne[1], 4, 1] .= nodes[2:(end - 1), Np[2], Np[3]]

                edgemap[1:Ne[1], 1, 2] .= nodes[(end - 1):-1:2, 1, 1]
                edgemap[1:Ne[1], 2, 2] .= nodes[(end - 1):-1:2, Np[2], 1]
                edgemap[1:Ne[1], 3, 2] .= nodes[(end - 1):-1:2, 1, Np[3]]
                edgemap[1:Ne[1], 4, 2] .= nodes[(end - 1):-1:2, Np[2], Np[3]]
            end

            if Np[2] > 2
                edgemap[1:Ne[2], 5, 1] .= nodes[1, 2:(end - 1), 1]
                edgemap[1:Ne[2], 6, 1] .= nodes[Np[1], 2:(end - 1), 1]
                edgemap[1:Ne[2], 7, 1] .= nodes[1, 2:(end - 1), Np[3]]
                edgemap[1:Ne[2], 8, 1] .= nodes[Np[1], 2:(end - 1), Np[3]]

                edgemap[1:Ne[2], 5, 2] .= nodes[1, (end - 1):-1:2, 1]
                edgemap[1:Ne[2], 6, 2] .= nodes[Np[1], (end - 1):-1:2, 1]
                edgemap[1:Ne[2], 7, 2] .= nodes[1, (end - 1):-1:2, Np[3]]
                edgemap[1:Ne[2], 8, 2] .= nodes[Np[1], (end - 1):-1:2, Np[3]]
            end

            if Np[3] > 2
                edgemap[1:Ne[3], 9, 1] .= nodes[1, 1, 2:(end - 1)]
                edgemap[1:Ne[3], 10, 1] .= nodes[Np[1], 1, 2:(end - 1)]
                edgemap[1:Ne[3], 11, 1] .= nodes[1, Np[2], 2:(end - 1)]
                edgemap[1:Ne[3], 12, 1] .= nodes[Np[1], Np[2], 2:(end - 1)]

                edgemap[1:Ne[3], 9, 2] .= nodes[1, 1, (end - 1):-1:2]
                edgemap[1:Ne[3], 10, 2] .= nodes[Np[1], 1, (end - 1):-1:2]
                edgemap[1:Ne[3], 11, 2] .= nodes[1, Np[2], (end - 1):-1:2]
                edgemap[1:Ne[3], 12, 2] .= nodes[Np[1], Np[2], (end - 1):-1:2]
            end
        else
            edgemap = nothing
        end

        Nf = Np .- 2
        Nf_max = maximum([Nf[1] * Nf[2], Nf[2] * Nf[3], Nf[1] * Nf[3]])
        if Nf_max ≥ 1
            facemap = -ones(Int64, Nf_max, 6, 2)

            if Nf[2] > 0 && Nf[3] > 0
                nfc = Nf[2] * Nf[3]
                facemap[1:nfc, 1, 1] .= nodes[1, 2:(end - 1), 2:(end - 1)][:]
                facemap[1:nfc, 2, 1] .=
                    nodes[Np[1], 2:(end - 1), 2:(end - 1)][:]

                facemap[1:nfc, 1, 2] .= nodes[1, (end - 1):-1:2, 2:(end - 1)][:]
                facemap[1:nfc, 2, 2] .=
                    nodes[Np[1], (end - 1):-1:2, 2:(end - 1)][:]
            end

            if Nf[1] > 0 && Nf[3] > 0
                nfc = Nf[1] * Nf[3]
                facemap[1:nfc, 3, 1] .= nodes[2:(end - 1), 1, 2:(end - 1)][:]
                facemap[1:nfc, 4, 1] .=
                    nodes[2:(end - 1), Np[2], 2:(end - 1)][:]

                facemap[1:nfc, 3, 2] .= nodes[(end - 1):-1:2, 1, 2:(end - 1)][:]
                facemap[1:nfc, 4, 2] .=
                    nodes[(end - 1):-1:2, Np[2], 2:(end - 1)][:]
            end

            if Nf[1] > 0 && Nf[2] > 0
                nfc = Nf[1] * Nf[2]
                facemap[1:nfc, 5, 1] .= nodes[2:(end - 1), 2:(end - 1), 1][:]
                facemap[1:nfc, 6, 1] .=
                    nodes[2:(end - 1), 2:(end - 1), Np[3]][:]

                facemap[1:nfc, 5, 2] .= nodes[(end - 1):-1:2, 2:(end - 1), 1][:]
                facemap[1:nfc, 6, 2] .=
                    nodes[(end - 1):-1:2, 2:(end - 1), Np[3]][:]
            end
        else
            facemap = nothing
        end
    else
        vertmap, edgemap, facemap = nothing, nothing, nothing
    end

    return vertmap, edgemap, facemap
end


"""
    commmapping(N, commelems, commfaces, nabrtocomm)

This function takes in a tuple of polynomial orders `N` and parts of a mesh (as
returned from `connectmesh` such as `sendelems`, `sendfaces`, and `nabrtosend`)
and returns index mappings for the element surface flux parallel communcation.
The returned `Tuple` contains:

 - `vmapC` an array of linear indices into the volume degrees of freedom to be
   communicated.

 - `nabrtovmapC` a range in `vmapC` to communicate with each neighbor.
"""
function commmapping(N, commelems, commfaces, nabrtocomm)
    nface, nelem = size(commfaces)

    @assert nelem == length(commelems)

    d = div(nface, 2)
    Nq = N .+ 1
    Np = prod(Nq)

    vmapC = similar(commelems, nelem * Np)
    nabrtovmapC = similar(nabrtocomm)

    i = 1
    e = 1
    for neighbor in 1:length(nabrtocomm)
        rbegin = i
        for ne in nabrtocomm[neighbor]
            ce = commelems[ne]

            # Whole element sending
            # for n = 1:Np
            #   vmapC[i] = (ce-1)*Np + n
            #   i += 1
            # end

            CI = CartesianIndices(ntuple(j -> 1:Nq[j], d))
            for (ci, li) in zip(CI, LinearIndices(CI))
                addpoint = false
                for j in 1:d
                    addpoint |=
                        (commfaces[2 * (j - 1) + 1, e] && ci[j] == 1) ||
                        (commfaces[2 * (j - 1) + 2, e] && ci[j] == Nq[j])
                end

                if addpoint
                    vmapC[i] = (ce - 1) * Np + li
                    i += 1
                end
            end

            e += 1
        end
        rend = i - 1

        nabrtovmapC[neighbor] = rbegin:rend
    end

    resize!(vmapC, i - 1)

    (vmapC, nabrtovmapC)
end

# Compute geometry FVM version
function computegeometry_fvm(elemtocoord, D, ξ, ω, meshwarp)
    FT = eltype(D[1])
    dim = length(D)
    nface = 2dim
    nelem = size(elemtocoord, 3)

    Nq = ntuple(j -> size(D[j], 1), dim)
    Np = prod(Nq)
    Nfp = div.(Np, Nq)

    Nq_N1 = max.(Nq, 2)
    Np_N1 = prod(Nq_N1)
    Nfp_N1 = div.(Np_N1, Nq_N1)

    # First we compute the geometry with all the N = 0 dimension set to N = 1
    # so that we can later compute the geometry for the case N = 0, as the
    # average of two N = 1 cases
    ξ1, ω1 = Elements.lglpoints(FT, 1)
    D1 = Elements.spectralderivative(ξ1)
    D_N1 = ntuple(j -> Nq[j] == 1 ? D1 : D[j], dim)
    ξ_N1 = ntuple(j -> Nq[j] == 1 ? ξ1 : ξ[j], dim)
    ω_N1 = ntuple(j -> Nq[j] == 1 ? ω1 : ω[j], dim)
    (vgeo_N1, sgeo_N1, x_vtk) =
        computegeometry(elemtocoord, D_N1, ξ_N1, ω_N1, meshwarp)

    # Allocate the storage for N = 0 volume metrics
    vgeo = VolumeGeometry(FT, Nq, nelem)

    # Counter to make sure we got all the vgeo terms
    num_vgeo_handled = 0

    Metrics.creategrid!(vgeo, elemtocoord, ξ)

    x1 = vgeo.x1
    x2 = vgeo.x2
    x3 = vgeo.x3
    @inbounds for j in 1:length(vgeo.x1)
        (x1[j], x2[j], x3[j]) = meshwarp(vgeo.x1[j], vgeo.x2[j], vgeo.x3[j])
    end

    # Update data in vgeo
    vgeo.x1 .= x1
    vgeo.x2 .= x2
    vgeo.x3 .= x3
    num_vgeo_handled += 3

    @views begin
        # ωJ should be a sum
        ωJ_N1 = reshape(vgeo_N1.ωJ, (Nq_N1..., nelem))
        vgeo.ωJ[:] .= sum(ωJ_N1, dims = findall(Nq .== 1))[:]
        num_vgeo_handled += 1

        # need to recompute ωJI
        vgeo.ωJI .= 1 ./ vgeo.ωJ
        num_vgeo_handled += 1

        # coordinates should just be averages
        avg_den = 2^sum(Nq .== 1)
        JcV_N1 = reshape(vgeo_N1.JcV, (Nq_N1..., nelem))
        vgeo.JcV[:] .= sum(JcV_N1, dims = findall(Nq .== 1))[:] ./ avg_den
        num_vgeo_handled += 1

        # For the metrics it is J * ξixk we approximate so multiply and divide the
        # mass matrix (which has the Jacobian determinant and the proper averaging
        # due to the quadrature weights)
        ωJ_N1 = reshape(vgeo_N1.ωJ, (Nq_N1..., nelem))
        ωJI = vgeo.ωJI

        ξ1x1_N1 = reshape(vgeo_N1.ξ1x1, (Nq_N1..., nelem))
        ξ2x1_N1 = reshape(vgeo_N1.ξ2x1, (Nq_N1..., nelem))
        ξ3x1_N1 = reshape(vgeo_N1.ξ3x1, (Nq_N1..., nelem))
        ξ1x2_N1 = reshape(vgeo_N1.ξ1x2, (Nq_N1..., nelem))
        ξ2x2_N1 = reshape(vgeo_N1.ξ2x2, (Nq_N1..., nelem))
        ξ3x2_N1 = reshape(vgeo_N1.ξ3x2, (Nq_N1..., nelem))
        ξ1x3_N1 = reshape(vgeo_N1.ξ1x3, (Nq_N1..., nelem))
        ξ2x3_N1 = reshape(vgeo_N1.ξ2x3, (Nq_N1..., nelem))
        ξ3x3_N1 = reshape(vgeo_N1.ξ3x3, (Nq_N1..., nelem))

        vgeo.ξ1x1[:] .=
            sum(ωJ_N1 .* ξ1x1_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ2x1[:] .=
            sum(ωJ_N1 .* ξ2x1_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ3x1[:] .=
            sum(ωJ_N1 .* ξ3x1_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ1x2[:] .=
            sum(ωJ_N1 .* ξ1x2_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ2x2[:] .=
            sum(ωJ_N1 .* ξ2x2_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ3x2[:] .=
            sum(ωJ_N1 .* ξ3x2_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ1x3[:] .=
            sum(ωJ_N1 .* ξ1x3_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ2x3[:] .=
            sum(ωJ_N1 .* ξ2x3_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        vgeo.ξ3x3[:] .=
            sum(ωJ_N1 .* ξ3x3_N1, dims = findall(Nq .== 1))[:] .* ωJI[:]
        num_vgeo_handled += 9

        # compute ωJH and JvC
        horizontal_metrics!(vgeo, Nq, ω)
        num_vgeo_handled += 1

        # Make sure we handled all the vgeo terms
        @assert _nvgeo == num_vgeo_handled
    end

    # Sort out the sgeo terms
    @views begin
        sgeo = SurfaceGeometry(FT, Nq, nface, nelem)

        # for the volume inverse mass matrix
        p = reshape(1:Np, Nq)
        if dim == 1
            fmask = (p[1:1], p[Nq[1]:Nq[1]])
        elseif dim == 2
            fmask = (p[1, :][:], p[Nq[1], :][:], p[:, 1][:], p[:, Nq[2]][:])
        elseif dim == 3
            fmask = (
                p[1, :, :][:],
                p[Nq[1], :, :][:],
                p[:, 1, :][:],
                p[:, Nq[2], :][:],
                p[:, :, 1][:],
                p[:, :, Nq[3]][:],
            )
        end
        for d in 1:dim
            for f in (2d - 1):(2d)
                # number of points matches means that we keep all the data
                # (N = 0 is not on the face)
                if Nfp[d] == Nfp_N1[d]
                    sgeo.n1[:, f, :] .= sgeo_N1.n1[:, f, :]
                    sgeo.n2[:, f, :] .= sgeo_N1.n2[:, f, :]
                    sgeo.n3[:, f, :] .= sgeo_N1.n3[:, f, :]
                    sgeo.sωJ[:, f, :] .= sgeo_N1.sωJ[:, f, :]

                    # Volume inverse mass will be wrong so reset it
                    sgeo.vωJI[:, f, :] .= vgeo.ωJI[fmask[f], :]
                else
                    # Counter to make sure we got all the sgeo terms
                    num_sgeo_handled = 0

                    # sum to get sM
                    Nq_f = (Nq[1:(d - 1)]..., Nq[(d + 1):dim]...)
                    Nq_f_N1 = (Nq_N1[1:(d - 1)]..., Nq_N1[(d + 1):dim]...)
                    sM_N1 = reshape(
                        sgeo_N1.sωJ[1:Nfp_N1[d], f, :],
                        Nq_f_N1...,
                        nelem,
                    )
                    sgeo.sωJ[1:Nfp[d], f, :][:] .=
                        sum(sM_N1, dims = findall(Nq_f .== 1))[:]
                    num_sgeo_handled += 1

                    # Normals (like metrics in the volume) need to be computed
                    # scaled by surface Jacobian which we can do with the
                    # surface mass matrices
                    sM = sgeo.sωJ[1:Nfp[d], f, :]

                    fld_N1_n1 = reshape(
                        sgeo_N1.n1[1:Nfp_N1[d], f, :],
                        Nq_f_N1...,
                        nelem,
                    )
                    fld_N1_n2 = reshape(
                        sgeo_N1.n2[1:Nfp_N1[d], f, :],
                        Nq_f_N1...,
                        nelem,
                    )
                    fld_N1_n3 = reshape(
                        sgeo_N1.n3[1:Nfp_N1[d], f, :],
                        Nq_f_N1...,
                        nelem,
                    )

                    sgeo.n1[1:Nfp[d], f, :][:] .=
                        sum(sM_N1 .* fld_N1_n1, dims = findall(Nq_f .== 1))[:] ./
                        sM[:]

                    sgeo.n2[1:Nfp[d], f, :][:] .=
                        sum(sM_N1 .* fld_N1_n2, dims = findall(Nq_f .== 1))[:] ./
                        sM[:]

                    sgeo.n3[1:Nfp[d], f, :][:] .=
                        sum(sM_N1 .* fld_N1_n3, dims = findall(Nq_f .== 1))[:] ./
                        sM[:]

                    num_sgeo_handled += 3

                    # set the volume inverse mass matrix
                    sgeo.vωJI[1:Nfp[d], f, :] .= vgeo.ωJI[fmask[f], :]
                    num_sgeo_handled += 1

                    # Make sure we handled all the vgeo terms
                    @assert _nsgeo == num_sgeo_handled
                end
            end
        end
    end

    (vgeo, sgeo, x_vtk)
end

"""
    computegeometry(elemtocoord, D, ξ, ω, meshwarp)

Compute the geometric factors data needed to define metric terms at
each quadrature point. First, compute the so called "topology coordinates"
from reference coordinates ξ. Then map these topology coordinate
to physical coordinates. Then compute the Jacobian of the mapping from
reference coordinates to physical coordinates, i.e., ∂x/∂ξ, by calling
`compute_reference_to_physical_coord_jacobian!`.
Finally, compute the metric terms by calling the function `computemetric!`.
"""
function computegeometry(elemtocoord, D, ξ, ω, meshwarp)
    FT = eltype(D[1])
    dim = length(D)
    nface = 2dim
    nelem = size(elemtocoord, 3)

    Nq = ntuple(j -> size(D[j], 1), dim)

    # Compute metric terms for FVM
    if any(Nq .== 1)
        return computegeometry_fvm(elemtocoord, D, ξ, ω, meshwarp)
    end

    Np = prod(Nq)
    Nfp = div.(Np, Nq)

    # Initialize volume and surface geometric term data structures
    vgeo = VolumeGeometry(FT, Nq, nelem)
    sgeo = SurfaceGeometry(FT, Nq, nface, nelem)

    # a) Compute "topology coordinates" from reference coordinates ξ
    Metrics.creategrid!(vgeo, elemtocoord, ξ)

    # Create local variables
    x1 = vgeo.x1
    x2 = vgeo.x2
    x3 = vgeo.x3

    # b) Map "topology coordinates" -> physical coordinates
    @inbounds for j in 1:length(vgeo.x1)
        (x1[j], x2[j], x3[j]) = meshwarp(vgeo.x1[j], vgeo.x2[j], vgeo.x3[j])
    end

    # Update global data in vgeo
    vgeo.x1 .= x1
    vgeo.x2 .= x2
    vgeo.x3 .= x3

    # c) Compute Jacobian matrix, ∂x/∂ξ
    Metrics.compute_reference_to_physical_coord_jacobian!(vgeo, nelem, D)

    # d) Compute the metric terms
    Metrics.computemetric!(vgeo, sgeo, D)

    # Note:
    # To get analytic derivatives, we need to be able differentiate through (a,b) and combine (a,b,c)

    # Compute the metric terms
    p = reshape(1:Np, Nq)
    if dim == 1
        fmask = (p[1:1], p[Nq[1]:Nq[1]])
    elseif dim == 2
        fmask = (p[1, :][:], p[Nq[1], :][:], p[:, 1][:], p[:, Nq[2]][:])
    elseif dim == 3
        fmask = (
            p[1, :, :][:],
            p[Nq[1], :, :][:],
            p[:, 1, :][:],
            p[:, Nq[2], :][:],
            p[:, :, 1][:],
            p[:, :, Nq[3]][:],
        )
    end

    # since `ξ1` is the fastest dimension and `ξdim` the slowest the tensor
    # product order is reversed
    M = kron(1, reverse(ω)...)
    vgeo.ωJ .*= M
    vgeo.ωJI .= 1 ./ vgeo.ωJ
    for d in 1:dim
        for f in (2d - 1):(2d)
            sgeo.vωJI[1:Nfp[d], f, :] .= vgeo.ωJI[fmask[f], :]
        end
    end

    sM = fill!(similar(sgeo.sωJ, maximum(Nfp), nface), NaN)
    for d in 1:dim
        for f in (2d - 1):(2d)
            ωf = ntuple(j -> ω[mod1(d + j, dim)], dim - 1)
            # Because of the `mod1` this face is already flipped
            if !(dim == 3 && d == 2)
                ωf = reverse(ωf)
            end
            sM[1:Nfp[d], f] .= dim > 1 ? kron(1, ωf...) : one(FT)
        end
    end
    sgeo.sωJ .*= sM

    # compute MH and JvC
    horizontal_metrics!(vgeo, Nq, ω)

    # This is mainly done to support FVM plotting when N=0 (since we need cell
    # edge values)
    x_vtk = (vgeo.x1, vgeo.x2, vgeo.x3)

    return (vgeo, sgeo, x_vtk)
end

"""
    horizontal_metrics!(vgeo::VolumeGeometry, Nq, ω)

Compute the horizontal mass matrix `ωJH` field of `vgeo`
```
J .* norm(∂ξ3/∂x) * (ωᵢ ⊗ ωⱼ); for integrating over a plane
```
(in 2-D ξ2 not ξ3 is used).
"""
function horizontal_metrics!(vgeo::VolumeGeometry, Nq, ω)
    dim = length(Nq)

    MH = dim == 1 ? 1 : kron(ones(1, Nq[dim]), reverse(ω[1:(dim - 1)])...)[:]
    M = vec(kron(1, reverse(ω)...))

    J = vgeo.ωJ ./ M

    # Compute |r'(ξ3)| for vertical line integrals
    if dim == 1
        vgeo.ωJH .= 1
    elseif dim == 2
        vgeo.ωJH .= MH .* hypot.(J .* vgeo.ξ2x1, J .* vgeo.ξ2x2)
    elseif dim == 3
        vgeo.ωJH .= MH .* hypot.(J .* vgeo.ξ3x1, J .* vgeo.ξ3x2, J .* vgeo.ξ3x3)
    else
        error("dim $dim not implemented")
    end
    return vgeo
end

"""
    indefinite_integral_interpolation_matrix(r, ω)

Given a set of integration points `r` and integration weights `ω` this computes
a matrix that will compute the indefinite integral of the (interpolant) of a
function and evaluate the indefinite integral at the points `r`.

Namely, let
```math
    q(ξ) = ∫_{ξ_{0}}^{ξ} f(ξ') dξ'
```
then we have that
```
I∫ * f.(r) = q.(r)
```
where `I∫` is the integration and interpolation matrix defined by this function.

!!! note

    The integration is done using the provided quadrature weight, so if these
    cannot integrate `f(ξ)` exactly, `f` is first interpolated and then
    integrated using quadrature. Namely, we have that:
    ```math
        q(ξ) = ∫_{ξ_{0}}^{ξ} I(f(ξ')) dξ'
    ```
    where `I` is the interpolation operator.

"""
function indefinite_integral_interpolation_matrix(r, ω)
    Nq = length(r)

    I∫ = similar(r, Nq, Nq)
    # first value is zero
    I∫[1, :] .= Nq == 1 ? ω[1] : 0

    # barycentric weights for interpolation
    wbary = Elements.baryweights(r)

    # Compute the interpolant of the indefinite integral
    for n in 2:Nq
        # grid from first dof to current point
        rdst = (1 .- r) / 2 * r[1] + (1 .+ r) / 2 * r[n]
        # interpolation matrix
        In = Elements.interpolationmatrix(r, rdst, wbary)
        # scaling from LGL to current of the interval
        Δ = (r[n] - r[1]) / 2
        # row of the matrix we have computed
        I∫[n, :] .= (Δ * ω' * In)[:]
    end
    I∫
end
# }}}

using KernelAbstractions.Extras: @unroll


const _x1 = Grids._x1
const _x2 = Grids._x2
const _x3 = Grids._x3
const _JcV = Grids._JcV

@doc """
    kernel_min_neighbor_distance!(::Val{N}, ::Val{dim}, direction,
                             min_neighbor_distance, vgeo, topology.realelems)

Computational kernel: Computes the minimum physical distance between node
neighbors within an element.

The `direction` in the reference element controls which nodes are considered
neighbors.
""" kernel_min_neighbor_distance!
@kernel function kernel_min_neighbor_distance!(
    ::Val{N},
    ::Val{dim},
    direction,
    min_neighbor_distance,
    vgeo,
    elems,
) where {N, dim}

    @uniform begin
        FT = eltype(min_neighbor_distance)
        Nq = N .+ 1
        Np = prod(Nq)

        if direction isa EveryDirection
            mininξ = (true, true, true)
        elseif direction isa HorizontalDirection
            mininξ = (true, dim == 2 ? false : true, false)
        elseif direction isa VerticalDirection
            mininξ = (false, dim == 2 ? true : false, dim == 2 ? false : true)
        end

        @inbounds begin
            # 2D Nq = (nh, nv)
            # 3D Nq = (nh, nh, nv)
            Nq1 = Nq[1]
            Nq2 = Nq[2]
            Nqk = dim == 2 ? 1 : Nq[end]
            mininξ1 = mininξ[1]
            mininξ2 = mininξ[2]
            mininξ3 = mininξ[3]
        end
    end


    I = @index(Global, Linear)
    # local element id
    e = (I - 1) ÷ Np + 1
    # local quadrature id
    ijk = (I - 1) % Np + 1

    # local i, j, k quadrature id
    i = (ijk - 1) % Nq1 + 1
    j = (ijk - 1) ÷ Nq1 % Nq2 + 1
    k = (ijk - 1) ÷ (Nq1 * Nq2) % Nqk + 1

    md = typemax(FT)

    x = SVector(vgeo[ijk, _x1, e], vgeo[ijk, _x2, e], vgeo[ijk, _x3, e])

    # first horizontal distance
    if mininξ1
        @unroll for î in (i - 1, i + 1)
            if 1 ≤ î ≤ Nq1
                îjk = î + Nq1 * (j - 1) + Nq1 * Nq2 * (k - 1)
                x̂ = SVector(
                    vgeo[îjk, _x1, e],
                    vgeo[îjk, _x2, e],
                    vgeo[îjk, _x3, e],
                )
                md = min(md, norm(x - x̂))
            end
        end
    end

    # second horizontal distance or vertical distance (dim=2)
    if mininξ2
        # FV Vercial direction, use 2vgeo[ijk, _JcV, e]
        if dim == 2 && Nq2 == 1
            md = min(md, 2vgeo[ijk, _JcV, e])
        else
            @unroll for ĵ in (j - 1, j + 1)
                if 1 ≤ ĵ ≤ Nq2
                    iĵk = i + Nq1 * (ĵ - 1) + Nq1 * Nq2 * (k - 1)
                    x̂ = SVector(
                        vgeo[iĵk, _x1, e],
                        vgeo[iĵk, _x2, e],
                        vgeo[iĵk, _x3, e],
                    )
                    md = min(md, norm(x - x̂))
                end
            end
        end
    end

    # vertical distance (dim=3)
    if mininξ3
        # FV Vercial direction, use 2vgeo[ijk, _JcV, e]
        if dim == 3 && Nqk == 1
            md = min(md, 2vgeo[ijk, _JcV, e])
        else
            @unroll for k̂ in (k - 1, k + 1)
                if 1 ≤ k̂ ≤ Nqk
                    ijk̂ = i + Nq1 * (j - 1) + Nq1 * Nq2 * (k̂ - 1)
                    x̂ = SVector(
                        vgeo[ijk̂, _x1, e],
                        vgeo[ijk̂, _x2, e],
                        vgeo[ijk̂, _x3, e],
                    )
                    md = min(md, norm(x - x̂))
                end
            end
        end
    end

    min_neighbor_distance[ijk, e] = md
end

struct QuadratureGrid{T, dim, N, Np, DA, DAT1, DAT2, DAT3} <:
       AbstractGrid{T, dim, N, Np, DA}

    "volume metric terms"
    vgeo::DAT3

    "1-D lgl weights on the device (one for each dimension)"
    ω::DAT1

    "1-D basis function matrix on the device (one for each dimension)"
    B::DAT2

    "Transpose 1-D basis function matrix on the device (one for each dimension)"
    Bᵀ::DAT2

    "1-D derivative operator on the device (one for each dimension)"
    D::DAT2

    "Transpose of 1-D derivative operator on the device (one for each dimension)"
    Dᵀ::DAT2

    "Temporary Storage on spectral element mesh"
    scratch::DAT3

    "Temporary Storage for fast tensor product"
    scratch_ftp::DAT3

end

function QuadratureGrid(
    grd::DiscontinuousSpectralElementGrid{T, dim, N, Np, DA},
    fac::T,
) where {T, dim, N, Np, DA}
    N_q = Int.(ceil.(N .* fac)) # quadrature degree
    Np_q = prod(N_q .+ 1)       # # of quadrature points
    # quadrature grid and weights
    ξωq = ntuple(
        j ->
            N_q[j] == 0 ? Elements.glpoints(T, N_q[j]) :
            Elements.lglpoints(T, N_q[j]),
        dim,
    )
    ξq, ωq = ntuple(j -> map(x -> x[j], ξωq), 2)
    # DG/CG p-grid and points
    ξω = ntuple(
        j ->
            N[j] == 0 ? Elements.glpoints(T, N[j]) :
            Elements.lglpoints(T, N[j]),
        dim,
    )
    ξ, ω = ntuple(j -> map(x -> x[j], ξω), 2)
    wb = Elements.baryweights.(ξ)

    B = ntuple(j -> Elements.interpolationmatrix(ξ[j], ξq[j], wb[j]), dim)
    Bᵀ = ntuple(j -> Array(transpose(B[j])), dim)
    D = ntuple(j -> B[j] * grd.D[j], dim)
    Dᵀ = ntuple(j -> Array(transpose(D[j])), dim)

    ωq = DA.(ωq)
    B = DA.(B)
    Bᵀ = DA.(Bᵀ)
    D = DA.(D)
    Dᵀ = DA.(Dᵀ)
    scratch = DA(Array{T}(undef, Np_q, 9, size(grd.scratch, 3)))
    scratch_ftp = DA(Array{T}(undef, Np_q, 3, size(grd.scratch, 3)))
    vgeo = DA(Array{T}(undef, Np_q, size(grd.vgeo, 2), size(grd.vgeo, 3)))

    computegeometry_quadmesh!(grd.vgeo, vgeo, B, ωq, DA)

    QuadratureGrid{T, dim, N, Np, DA, typeof(ωq), typeof(B), typeof(scratch)}(
        vgeo,
        ωq,
        B,
        Bᵀ,
        D,
        Dᵀ,
        scratch,
        scratch_ftp,
    )
end

function computegeometry_quadmesh!(
    vgeo::FTA3D,
    vgeo_q::FTA3D,
    B_q::Tuple{FTA2D, FTA2D, FTA2D},
    ω_q::Tuple{FTA1D, FTA1D, FTA1D},
    ::Type{DA},
) where {
    FT <: AbstractFloat,
    FTA1D <: AbstractArray{FT, 1},
    FTA2D <: AbstractArray{FT, 2},
    FTA3D <: AbstractArray{FT, 3},
    DA,
}
    si, sj, sk = size(B_q[1], 2), size(B_q[2], 2), size(B_q[3], 2)
    sr, ss, st = size(B_q[1], 1), size(B_q[2], 1), size(B_q[3], 1)
    dims = (si, sj, sk, sr, ss, st)
    Nel = size(vgeo, 3)
    max_threads = 256
    for i in (
        _ξ1x1,
        _ξ2x1,
        _ξ3x1,
        _ξ1x2,
        _ξ2x2,
        _ξ3x2,
        _ξ1x3,
        _ξ2x3,
        _ξ3x3,
        _x1,
        _x2,
        _x3,
    )
        tpxv!(
            view(vgeo, :, i, :),
            nothing,
            view(vgeo_q, :, i, :),
            B_q[1],
            B_q[2],
            B_q[3],
            Val(dims),
            max_threads,
        )
    end
    # computing wt * jac on over-integration grid
    ξ1x1 = view(vgeo_q, :, _ξ1x1, :)
    ξ2x1 = view(vgeo_q, :, _ξ2x1, :)
    ξ3x1 = view(vgeo_q, :, _ξ3x1, :)
    ξ1x2 = view(vgeo_q, :, _ξ1x2, :)
    ξ2x2 = view(vgeo_q, :, _ξ2x2, :)
    ξ3x2 = view(vgeo_q, :, _ξ3x2, :)
    ξ1x3 = view(vgeo_q, :, _ξ1x3, :)
    ξ2x3 = view(vgeo_q, :, _ξ2x3, :)
    ξ3x3 = view(vgeo_q, :, _ξ3x3, :)
    wjac = view(vgeo_q, :, _M, :)

    wjac .=
        FT(1) ./ (
            ξ1x1 .* (ξ2x2 .* ξ3x3 .- ξ2x3 .* ξ3x2) .-
            ξ1x2 .* (ξ2x1 .* ξ3x3 .- ξ2x3 .* ξ3x1) .+
            ξ1x3 .* (ξ2x1 .* ξ3x2 .- ξ2x2 .* ξ3x1)
        )

    wq = DA(kron(1, reverse(Array.(ω_q))...))

    for i in 1:Nel
        wjac[:, i] .*= wq
    end

    return nothing
end

end # module
