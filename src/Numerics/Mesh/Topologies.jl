module Topologies
using ClimateMachine
using CubedSphere, Rotations
import ..BrickMesh
import MPI
using CUDA
using DocStringExtensions

export AbstractTopology,
    BrickTopology,
    StackedBrickTopology,
    CubedShellTopology,
    StackedCubedSphereTopology,
    AnalyticalTopography,
    NoTopography,
    DCMIPMountain,
    EquidistantCubedSphere,
    EquiangularCubedSphere,
    isstacked,
    cubed_sphere_topo_warp,
    compute_lat_long,
    compute_analytical_topography,
    cubed_sphere_warp,
    cubed_sphere_unwarp,
    equiangular_cubed_sphere_warp,
    equiangular_cubed_sphere_unwarp,
    equidistant_cubed_sphere_warp,
    equidistant_cubed_sphere_unwarp,
    conformal_cubed_sphere_warp

export grid1d, SingleExponentialStretching, InteriorStretching

export basic_topology_info

"""
    AbstractTopology{dim, T, nb}

Represents the connectivity of individual elements, with local dimension `dim`
with `nb` boundary conditions types. The element coordinates are of type `T`.
"""
abstract type AbstractTopology{dim, T, nb} end

"""
    BoxElementTopology{dim, T, nb} <: AbstractTopology{dim,T,nb}

The local topology of a larger MPI-distributed topology, represented by
`dim`-dimensional box elements, with `nb` boundary conditions.

This contains the necessary information for the connectivity elements of the
elements on the local process, along with "ghost" elements from neighbouring
processes.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct BoxElementTopology{dim, T, nb} <: AbstractTopology{dim, T, nb}

    """
    MPI communicator for communicating with neighbouring processes.
    """
    mpicomm::MPI.Comm

    """
    Range of element indices
    """
    elems::UnitRange{Int64}

    """
    Range of real (aka nonghost) element indices
    """
    realelems::UnitRange{Int64}

    """
    Range of ghost element indices
    """
    ghostelems::UnitRange{Int64}

    """
    Ghost element to face is received; `ghostfaces[f,ge] == true` if face `f` of
    ghost element `ge` is received.
    """
    ghostfaces::BitArray{2}

    """
    Array of send element indices
    """
    sendelems::Array{Int64, 1}

    """
    Send element to face is sent; `sendfaces[f,se] == true` if face `f` of send
    element `se` is sent.
    """
    sendfaces::BitArray{2}

    """
    Array of real elements that do not have a ghost element as a neighbor.
    """
    interiorelems::Array{Int64, 1}

    """
    Array of real elements that have at least on ghost element as a neighbor.

    Note that this is different from `sendelems` because `sendelems` duplicates
    elements that need to be sent to multiple neighboring processes.
    """
    exteriorelems::Array{Int64, 1}

    """
    Element to vertex coordinates; `elemtocoord[d,i,e]` is the `d`th coordinate
    of corner `i` of element `e`

    !!! note
        currently coordinates always are of size 3 for `(x1, x2, x3)`
    """
    elemtocoord::Array{T, 3}

    """
    Element to neighboring element; `elemtoelem[f,e]` is the number of the
    element neighboring element `e` across face `f`. If it is a boundary face,
    then it is the boundary element index.
    """
    elemtoelem::Array{Int64, 2}

    """
    Element to neighboring element face; `elemtoface[f,e]` is the face number of
    the element neighboring element `e` across face `f`.  If there is no
    neighboring element then `elemtoface[f,e] == f`."
    """
    elemtoface::Array{Int64, 2}

    """
    element to neighboring element order; `elemtoordr[f,e]` is the ordering
    number of the element neighboring element `e` across face `f`.  If there is
    no neighboring element then `elemtoordr[f,e] == 1`.
    """
    elemtoordr::Array{Int64, 2}

    """
    Element to boundary number; `elemtobndy[f,e]` is the boundary number of face
    `f` of element `e`.  If there is a neighboring element then `elemtobndy[f,e]
    == 0`.
    """
    elemtobndy::Array{Int64, 2}

    """
    List of the MPI ranks for the neighboring processes
    """
    nabrtorank::Array{Int64, 1}

    """
    Range in ghost elements to receive for each neighbor
    """
    nabrtorecv::Array{UnitRange{Int64}, 1}

    """
    Range in `sendelems` to send for each neighbor
    """
    nabrtosend::Array{UnitRange{Int64}, 1}

    """
    original order in partitioning
    """
    origsendorder::Array{Int64, 1}

    """
    Tuple of boundary to element. `bndytoelem[b][i]` is the element which faces
    the `i`th boundary element of boundary `b`.
    """
    bndytoelem::NTuple{nb, Array{Int64, 1}}

    """
    Tuple of boundary to element face. `bndytoface[b][i]` is the face number of
    the element which faces the `i`th boundary element of boundary `b`.
    """
    bndytoface::NTuple{nb, Array{Int64, 1}}

    """
    Element to locally unique vertex number; `elemtouvert[v,e]` is the `v`th vertex
    of element `e`
    """
    elemtouvert::Union{Array{Int64, 2}, Nothing}

    """
    Vertex connectivity information for direct stiffness summation;
    `vtconn[vtconnoff[i]:vtconnoff[i+1]-1] == vtconn[lvt1, lelem1, lvt2, lelem2, ....]`
    for each rank-local unique vertex number,
    where `lvt1` is the element-local vertex number of element `lelem1`, etc.
    Vertices, not shared by multiple elements, are ignored.
    """
    vtconn::Union{AbstractArray{Int64, 1}, Nothing}

    """
    Vertex offset for vertex connectivity information
    """
    vtconnoff::Union{AbstractArray{Int64, 1}, Nothing}

    """
    Face connectivity information for direct stiffness summation;
    `fcconn[ufc,:] = [lfc1, lelem1, lfc2, lelem2, ordr]`
    where `ufc` is the rank-local unique face number, and
    `lfc1` is the element-local face number of element `lelem1`, etc.,
    and ordr is the relative orientation. Faces, not shared by multiple
    elements are ignored.
    """
    fcconn::Union{AbstractArray{Int64, 2}, Nothing}

    """
    Edge connectivity information for direct stiffness summation;
    `edgconn[edgconnoff[i]:edgconnoff[i+1]-1] == edgconn[ledg1, orient, lelem1, ledg2, orient, lelem2, ...]`
    for each rank-local unique edge number,
    where `ledg1` is the element-local edge number, `orient1` is orientation (forward/reverse) of dof along edge.
    Edges, not shared by multiple elements, are ignored.
    """
    edgconn::Union{AbstractArray{Int64, 1}, Nothing}

    """
    Edge offset for edge connectivity information
    """
    edgconnoff::Union{AbstractArray{Int64, 1}, Nothing}

    function BoxElementTopology{dim, T, nb}(
        mpicomm,
        elems,
        realelems,
        ghostelems,
        ghostfaces,
        sendelems,
        sendfaces,
        elemtocoord,
        elemtoelem,
        elemtoface,
        elemtoordr,
        elemtobndy,
        nabrtorank,
        nabrtorecv,
        nabrtosend,
        origsendorder,
        bndytoelem,
        bndytoface;
        device_array = ClimateMachine.array_type(),
        elemtouvert = nothing,
        vtconn = nothing,
        vtconnoff = nothing,
        fcconn = nothing,
        edgconn = nothing,
        edgconnoff = nothing,
    ) where {dim, T, nb}

        exteriorelems = sort(unique(sendelems))
        interiorelems = sort(setdiff(realelems, exteriorelems))
        if vtconn isa Array && vtconnoff isa Array
            vtconn = device_array(vtconn)
            vtconnoff = device_array(vtconnoff)
        end
        if fcconn isa Array
            fcconn = device_array(fcconn)
        end
        if edgconn isa Array && edgconnoff isa Array
            edgconn = device_array(edgconn)
            edgconnoff = device_array(edgconnoff)
        end
        return new{dim, T, nb}(
            mpicomm,
            elems,
            realelems,
            ghostelems,
            ghostfaces,
            sendelems,
            sendfaces,
            interiorelems,
            exteriorelems,
            elemtocoord,
            elemtoelem,
            elemtoface,
            elemtoordr,
            elemtobndy,
            nabrtorank,
            nabrtorecv,
            nabrtosend,
            origsendorder,
            bndytoelem,
            bndytoface,
            elemtouvert,
            vtconn,
            vtconnoff,
            fcconn,
            edgconn,
            edgconnoff,
        )
    end
end

"""
    hasboundary(topology::AbstractTopology)

query function to check whether a topology has a boundary (i.e., not fully
periodic)
"""
hasboundary(topology::AbstractTopology{dim, T, nb}) where {dim, T, nb} = nb != 0

if VERSION >= v"1.2-"
    isstacked(::T) where {T <: AbstractTopology} = hasfield(T, :stacksize)
else
    isstacked(::T) where {T <: AbstractTopology} =
        Base.fieldindex(T, :stacksize, false) > 0
end

"""
    BrickTopology{dim, T, nb} <: AbstractTopology{dim, T, nb}

A simple grid-based topology. This is a convenience wrapper around
[`BoxElementTopology`](@ref).
"""
struct BrickTopology{dim, T, nb} <: AbstractTopology{dim, T, nb}
    topology::BoxElementTopology{dim, T, nb}
end
Base.getproperty(a::BrickTopology, p::Symbol) =
    getproperty(getfield(a, :topology), p)

"""
    CubedShellTopology{T} <: AbstractTopology{2, T, 0}

A cube-shell topology. This is a convenience wrapper around
[`BoxElementTopology`](@ref).
"""
struct CubedShellTopology{T} <: AbstractTopology{2, T, 0}
    topology::BoxElementTopology{2, T, 0}
end
Base.getproperty(a::CubedShellTopology, p::Symbol) =
    getproperty(getfield(a, :topology), p)

abstract type AbstractStackedTopology{dim, T, nb} <:
              AbstractTopology{dim, T, nb} end


"""
    StackedBrickTopology{dim, T, nb} <: AbstractStackedTopology{dim}

A simple grid-based topology, where all elements on the trailing dimension are
stacked to be contiguous. This is a convenience wrapper around
[`BoxElementTopology`](@ref).
"""
struct StackedBrickTopology{dim, T, nb} <: AbstractStackedTopology{dim, T, nb}
    topology::BoxElementTopology{dim, T, nb}
    stacksize::Int64
    periodicstack::Bool
end
function Base.getproperty(a::StackedBrickTopology, p::Symbol)
    return p in (:stacksize, :periodicstack) ? getfield(a, p) :
           getproperty(getfield(a, :topology), p)
end

"""
    StackedCubedSphereTopology{T, nb} <: AbstractStackedTopology{3, T, nb}

A cube-sphere topology. All elements on the same "vertical" dimension are
stacked to be contiguous. This is a convenience wrapper around
[`BoxElementTopology`](@ref).
"""
struct StackedCubedSphereTopology{T, nb} <: AbstractStackedTopology{3, T, nb}
    topology::BoxElementTopology{3, T, nb}
    stacksize::Int64
end
function Base.getproperty(a::StackedCubedSphereTopology, p::Symbol)
    if p == :periodicstack
        return false
    else
        return p == :stacksize ? getfield(a, p) :
               getproperty(getfield(a, :topology), p)
    end
end

""" A wrapper for the BrickTopology """
BrickTopology(mpicomm, Nelems::NTuple{N, Integer}; kw...) where {N} =
    BrickTopology(mpicomm, map(Ne -> 0:Ne, Nelems); kw...)

"""
    BrickTopology{dim, T}(mpicomm, elemrange; boundary, periodicity)

Generate a brick mesh topology with coordinates given by the tuple `elemrange`
and the periodic dimensions given by the `periodicity` tuple.

The elements of the brick are partitioned equally across the MPI ranks based
on a space-filling curve.

By default boundary faces will be marked with a one and other faces with a
zero.  Specific boundary numbers can also be passed for each face of the brick
in `boundary`.  This will mark the nonperiodic brick faces with the given
boundary number.

# Examples

We can build a 3 by 2 element two-dimensional mesh that is periodic in the
\$x2\$-direction with
```jldoctest brickmesh

using ClimateMachine.Topologies
using MPI
MPI.Init()
topology = BrickTopology(MPI.COMM_SELF, (2:5,4:6);
                         periodicity=(false,true),
                         boundary=((1,2),(3,4)))
```
This returns the mesh structure for

             x2

              ^
              |
             6-  +-----+-----+-----+
              |  |     |     |     |
              |  |  3  |  4  |  5  |
              |  |     |     |     |
             5-  +-----+-----+-----+
              |  |     |     |     |
              |  |  1  |  2  |  6  |
              |  |     |     |     |
             4-  +-----+-----+-----+
              |
              +--|-----|-----|-----|--> x1
                 2     3     4     5

For example, the (dimension by number of corners by number of elements) array
`elemtocoord` gives the coordinates of the corners of each element.
```jldoctest brickmesh
julia> topology.elemtocoord
2×4×6 Array{Int64,3}:
[:, :, 1] =
 2  3  2  3
 4  4  5  5

[:, :, 2] =
 3  4  3  4
 4  4  5  5

[:, :, 3] =
 2  3  2  3
 5  5  6  6

[:, :, 4] =
 3  4  3  4
 5  5  6  6

[:, :, 5] =
 4  5  4  5
 5  5  6  6

[:, :, 6] =
 4  5  4  5
 4  4  5  5
```
Note that the corners are listed in Cartesian order.

The (number of faces by number of elements) array `elemtobndy` gives the
boundary number for each face of each element.  A zero will be given for
connected faces.
```jldoctest brickmesh
julia> topology.elemtobndy
4×6 Array{Int64,2}:
 1  0  1  0  0  0
 0  0  0  0  2  2
 0  0  0  0  0  0
 0  0  0  0  0  0
```
Note that the faces are listed in Cartesian order.

"""
function BrickTopology(
    mpicomm,
    elemrange;
    boundary = ntuple(j -> (1, 1), length(elemrange)),
    periodicity = ntuple(j -> false, length(elemrange)),
    connectivity = :face,
    ghostsize = 1,
)

    if boundary isa Matrix
        boundary = tuple(mapslices(x -> tuple(x...), boundary, dims = 1)...)
    end


    # We cannot handle anything else right now...
    @assert ghostsize == 1

    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    topology = BrickMesh.brickmesh(
        elemrange,
        periodicity,
        part = mpirank + 1,
        numparts = mpisize,
        boundary = boundary,
    )
    topology = BrickMesh.partition(mpicomm, topology...)
    origsendorder = topology[5]
    topology =
        connectivity == :face ?
        BrickMesh.connectmesh(mpicomm, topology[1:4]...) :
        BrickMesh.connectmeshfull(mpicomm, topology[1:4]...)
    bndytoelem, bndytoface = BrickMesh.enumerateboundaryfaces!(
        topology.elemtoelem,
        topology.elemtobndy,
        periodicity,
        boundary,
    )

    nb = length(bndytoelem)
    dim = length(elemrange)
    T = eltype(topology.elemtocoord)
    return BrickTopology{dim, T, nb}(BoxElementTopology{dim, T, nb}(
        mpicomm,
        topology.elems,
        topology.realelems,
        topology.ghostelems,
        topology.ghostfaces,
        topology.sendelems,
        topology.sendfaces,
        topology.elemtocoord,
        topology.elemtoelem,
        topology.elemtoface,
        topology.elemtoordr,
        topology.elemtobndy,
        topology.nabrtorank,
        topology.nabrtorecv,
        topology.nabrtosend,
        origsendorder,
        bndytoelem,
        bndytoface,
        elemtouvert = topology.elemtovert,
    ))
end

""" A wrapper for the StackedBrickTopology """
StackedBrickTopology(mpicomm, Nelems::NTuple{N, Integer}; kw...) where {N} =
    StackedBrickTopology(mpicomm, map(Ne -> 0:Ne, Nelems); kw...)

"""
    StackedBrickTopology{dim, T}(mpicomm, elemrange; boundary, periodicity)

Generate a stacked brick mesh topology with coordinates given by the tuple
`elemrange` and the periodic dimensions given by the `periodicity` tuple.

The elements are stacked such that the elements associated with range
`elemrange[dim]` are contiguous in the element ordering.

The elements of the brick are partitioned equally across the MPI ranks based
on a space-filling curve.  Further, stacks are not split at MPI boundaries.

By default boundary faces will be marked with a one and other faces with a
zero.  Specific boundary numbers can also be passed for each face of the brick
in `boundary`.  This will mark the nonperiodic brick faces with the given
boundary number.

# Examples

We can build a 3 by 2 element two-dimensional mesh that is periodic in the
\$x2\$-direction with
```jldoctest brickmesh

using ClimateMachine.Topologies
using MPI
MPI.Init()
topology = StackedBrickTopology(MPI.COMM_SELF, (2:5,4:6);
                                periodicity=(false,true),
                                boundary=((1,2),(3,4)))
```
This returns the mesh structure stacked in the \$x2\$-direction for

             x2

              ^
              |
             6-  +-----+-----+-----+
              |  |     |     |     |
              |  |  2  |  4  |  6  |
              |  |     |     |     |
             5-  +-----+-----+-----+
              |  |     |     |     |
              |  |  1  |  3  |  5  |
              |  |     |     |     |
             4-  +-----+-----+-----+
              |
              +--|-----|-----|-----|--> x1
                 2     3     4     5

For example, the (dimension by number of corners by number of elements) array
`elemtocoord` gives the coordinates of the corners of each element.
```jldoctest brickmesh
julia> topology.elemtocoord
2×4×6 Array{Int64,3}:
[:, :, 1] =
 2  3  2  3
 4  4  5  5

[:, :, 2] =
 2  3  2  3
 5  5  6  6

[:, :, 3] =
 3  4  3  4
 4  4  5  5

[:, :, 4] =
 3  4  3  4
 5  5  6  6

[:, :, 5] =
 4  5  4  5
 4  4  5  5

[:, :, 6] =
 4  5  4  5
 5  5  6  6
```
Note that the corners are listed in Cartesian order.

The (number of faces by number of elements) array `elemtobndy` gives the
boundary number for each face of each element.  A zero will be given for
connected faces.
```jldoctest brickmesh
julia> topology.elemtobndy
4×6 Array{Int64,2}:
 1  0  1  0  0  0
 0  0  0  0  2  2
 0  0  0  0  0  0
 0  0  0  0  0  0
```
Note that the faces are listed in Cartesian order.
"""
function StackedBrickTopology(
    mpicomm,
    elemrange;
    boundary = ntuple(j -> (1, 1), length(elemrange)),
    periodicity = ntuple(j -> false, length(elemrange)),
    connectivity = :full,
    ghostsize = 1,
)

    if boundary isa Matrix
        boundary = tuple(mapslices(x -> tuple(x...), boundary, dims = 1)...)
    end

    dim = length(elemrange)

    dim <= 1 && error("Stacked brick topology works for 2D and 3D")

    # Build the base topology
    basetopo = BrickTopology(
        mpicomm,
        elemrange[1:(dim - 1)];
        boundary = boundary[1:(dim - 1)],
        periodicity = periodicity[1:(dim - 1)],
        connectivity = connectivity,
        ghostsize = ghostsize,
    )


    # Use the base topology to build the stacked topology
    stack = elemrange[dim]
    stacksize = length(stack) - 1

    nvert = 2^dim
    nface = 2dim

    nreal = length(basetopo.realelems) * stacksize
    nghost = length(basetopo.ghostelems) * stacksize

    elems = 1:(nreal + nghost)
    realelems = 1:nreal
    ghostelems = nreal .+ (1:nghost)

    sendelems =
        similar(basetopo.sendelems, length(basetopo.sendelems) * stacksize)
    for i in 1:length(basetopo.sendelems), j in 1:stacksize
        sendelems[stacksize * (i - 1) + j] =
            stacksize * (basetopo.sendelems[i] - 1) + j
    end

    ghostfaces = similar(basetopo.ghostfaces, nface, length(ghostelems))
    ghostfaces .= false

    for i in 1:length(basetopo.ghostelems), j in 1:stacksize
        e = stacksize * (i - 1) + j
        for f in 1:(2 * (dim - 1))
            ghostfaces[f, e] = basetopo.ghostfaces[f, i]
        end
    end

    sendfaces = similar(basetopo.sendfaces, nface, length(sendelems))
    sendfaces .= false

    for i in 1:length(basetopo.sendelems), j in 1:stacksize
        e = stacksize * (i - 1) + j
        for f in 1:(2 * (dim - 1))
            sendfaces[f, e] = basetopo.sendfaces[f, i]
        end
    end

    elemtocoord = similar(basetopo.elemtocoord, dim, nvert, length(elems))

    for i in 1:length(basetopo.elems), j in 1:stacksize
        e = stacksize * (i - 1) + j

        for v in 1:(2^(dim - 1))
            for d in 1:(dim - 1)
                elemtocoord[d, v, e] = basetopo.elemtocoord[d, v, i]
                elemtocoord[d, 2^(dim - 1) + v, e] =
                    basetopo.elemtocoord[d, v, i]
            end

            elemtocoord[dim, v, e] = stack[j]
            elemtocoord[dim, 2^(dim - 1) + v, e] = stack[j + 1]
        end
    end

    elemtoelem = similar(basetopo.elemtoelem, nface, length(elems))
    elemtoface = similar(basetopo.elemtoface, nface, length(elems))
    elemtoordr = similar(basetopo.elemtoordr, nface, length(elems))
    elemtobndy = similar(basetopo.elemtobndy, nface, length(elems))

    for e in 1:(length(basetopo.elems) * stacksize), f in 1:nface
        elemtoelem[f, e] = e
        elemtoface[f, e] = f
        elemtoordr[f, e] = 1
        elemtobndy[f, e] = 0
    end

    for i in 1:length(basetopo.realelems), j in 1:stacksize
        e1 = stacksize * (i - 1) + j

        for f in 1:(2 * (dim - 1))
            e2 = stacksize * (basetopo.elemtoelem[f, i] - 1) + j

            elemtoelem[f, e1] = e2
            elemtoface[f, e1] = basetopo.elemtoface[f, i]

            # We assume a simple orientation right now
            @assert basetopo.elemtoordr[f, i] == 1
            elemtoordr[f, e1] = basetopo.elemtoordr[f, i]
        end

        et = stacksize * (i - 1) + j + 1
        eb = stacksize * (i - 1) + j - 1
        ft = 2 * (dim - 1) + 1
        fb = 2 * (dim - 1) + 2
        ot = 1
        ob = 1

        if j == stacksize
            et = periodicity[dim] ? stacksize * (i - 1) + 1 : e1
            ft = periodicity[dim] ? ft : 2 * (dim - 1) + 2
        end
        if j == 1
            eb = periodicity[dim] ? stacksize * (i - 1) + stacksize : e1
            fb = periodicity[dim] ? fb : 2 * (dim - 1) + 1
        end

        elemtoelem[2 * (dim - 1) + 1, e1] = eb
        elemtoelem[2 * (dim - 1) + 2, e1] = et
        elemtoface[2 * (dim - 1) + 1, e1] = fb
        elemtoface[2 * (dim - 1) + 2, e1] = ft
        elemtoordr[2 * (dim - 1) + 1, e1] = ob
        elemtoordr[2 * (dim - 1) + 2, e1] = ot
    end

    for i in 1:length(basetopo.elems), j in 1:stacksize
        e1 = stacksize * (i - 1) + j

        for f in 1:(2 * (dim - 1))
            elemtobndy[f, e1] = basetopo.elemtobndy[f, i]
        end

        bt = bb = 0

        if j == stacksize
            bt = periodicity[dim] ? bt : boundary[dim][2]
        end
        if j == 1
            bb = periodicity[dim] ? bb : boundary[dim][1]
        end

        elemtobndy[2 * (dim - 1) + 1, e1] = bb
        elemtobndy[2 * (dim - 1) + 2, e1] = bt
    end

    nabrtorank = basetopo.nabrtorank
    nabrtorecv = UnitRange{Int}[
        UnitRange(
            stacksize * (first(basetopo.nabrtorecv[n]) - 1) + 1,
            stacksize * last(basetopo.nabrtorecv[n]),
        ) for n in 1:length(nabrtorank)
    ]
    nabrtosend = UnitRange{Int}[
        UnitRange(
            stacksize * (first(basetopo.nabrtosend[n]) - 1) + 1,
            stacksize * last(basetopo.nabrtosend[n]),
        ) for n in 1:length(nabrtorank)
    ]

    bndytoelem, bndytoface = BrickMesh.enumerateboundaryfaces!(
        elemtoelem,
        elemtobndy,
        periodicity,
        boundary,
    )
    nb = length(bndytoelem)

    T = eltype(basetopo.elemtocoord)
    #----setting up DSS----------
    if basetopo.elemtouvert isa Array
        #---setup vertex DSS
        nvertby2 = div(nvert, 2)
        elemtouvert = similar(basetopo.elemtouvert, nvert, length(elems))
        # base level
        for i in 1:length(basetopo.elems)
            e = stacksize * (i - 1) + 1
            for vt in 1:nvertby2
                elemtouvert[vt, e] =
                    (basetopo.elemtouvert[vt, i] - 1) * (stacksize + 1) + 1
            end
        end
        # interior levels
        for i in 1:length(basetopo.elems), j in 1:(stacksize - 1)
            e = stacksize * (i - 1) + j
            for vt in 1:nvertby2
                elemtouvert[nvertby2 + vt, e] = elemtouvert[vt, e] + 1
                elemtouvert[vt, e + 1] = elemtouvert[nvertby2 + vt, e]
            end
        end
        # top level
        if periodicity[dim]
            for i in 1:length(basetopo.elems)
                e = stacksize * i
                for vt in 1:nvertby2
                    elemtouvert[nvertby2 + vt, e] =
                        (basetopo.elemtouvert[vt, i] - 1) * (stacksize + 1) + 1
                end
            end
        else
            for i in 1:length(basetopo.elems)
                e = stacksize * i
                for vt in 1:nvertby2
                    elemtouvert[nvertby2 + vt, e] = elemtouvert[vt, e] + 1
                end
            end
        end
        vtmax = maximum(unique(elemtouvert))
        vtconn = map(j -> zeros(Int, j), zeros(Int, vtmax))
        for el in elems, lvt in 1:nvert
            vt = elemtouvert[lvt, el]
            push!(vtconn[vt], lvt) # local vertex number
            push!(vtconn[vt], el)  # local elem number
        end
        # building the vertex connectivity device array
        vtconnoff = zeros(Int, vtmax + 1)
        vtconnoff[1] = 1
        temp = Int64[]
        for vt in 1:vtmax
            nconn = length(vtconn[vt])
            if nconn > 2
                vtconnoff[vt + 1] = vtconnoff[vt] + nconn
                append!(temp, vtconn[vt])
            else
                vtconnoff[vt + 1] = vtconnoff[vt]
            end
        end
        vtconn = temp
        #---setup face DSS---------------------
        fcmarker = -ones(Int, nface, length(elems))
        fcno = 0
        for el in realelems
            for fc in 1:nface
                if elemtobndy[fc, el] == 0
                    nabrel = elemtoelem[fc, el]
                    nabrfc = elemtoface[fc, el]
                    if fcmarker[fc, el] == -1 && fcmarker[nabrfc, nabrel] == -1
                        fcno += 1
                        fcmarker[fc, el] = fcno
                        fcmarker[nabrfc, nabrel] = fcno
                    end
                end
            end
        end
        fcconn = -ones(Int, fcno, 5)
        for el in realelems
            for fc in 1:nface
                fcno = fcmarker[fc, el]
                ordr = elemtoordr[fc, el]
                if fcno ≠ -1
                    nabrel = elemtoelem[fc, el]
                    nabrfc = elemtoface[fc, el]
                    fcconn[fcno, 1] = fc
                    fcconn[fcno, 2] = el
                    fcconn[fcno, 3] = nabrfc
                    fcconn[fcno, 4] = nabrel
                    fcconn[fcno, 5] = ordr
                    fcmarker[fc, el] = -1
                    fcmarker[nabrfc, nabrel] = -1
                end
            end
        end
        #---setup edge DSS---------------------
        if dim == 3
            nedge = 12
            edgemask = [
                1 3 5 7 1 2 5 6 1 2 3 4
                2 4 6 8 3 4 7 8 5 6 7 8
            ]
            edges = Array{Int64}(undef, 6, nedge * length(elems))
            ledge = zeros(Int64, 2)
            uedge = Dict{Array{Int64, 1}, Int64}()
            orient = 1
            uedgno = Int64(0)
            ctr = 1
            for el in elems
                for edg in 1:nedge
                    orient = 1
                    for i in 1:2
                        ledge[i] = elemtouvert[edgemask[i, edg], el]
                        edges[i, ctr] = ledge[i]                     # edge vertices [1:2]
                    end
                    if ledge[1] > ledge[2]
                        sort!(ledge)
                        orient = 2
                    end
                    edges[3, ctr] = edg    # local edge number
                    edges[4, ctr] = orient # edge orientation
                    edges[5, ctr] = el     # element number
                    if haskey(uedge, ledge[:])
                        edges[6, ctr] = uedge[ledge[:]]  # unique edge number
                    else
                        uedgno += 1
                        uedge[ledge[:]] = uedgno
                        edges[6, ctr] = uedgno
                    end
                    ctr += 1
                end
            end
            edgconn = map(j -> zeros(Int, j), zeros(Int, uedgno))
            ctr = 1
            for el in elems
                for edg in 1:nedge
                    uedg = edges[6, ctr]
                    push!(edgconn[uedg], edg)           # local edge number
                    push!(edgconn[uedg], edges[4, ctr]) # orientation
                    push!(edgconn[uedg], el)            # element number
                    ctr += 1
                end
            end
            # remove edges belonging to a single element and edges
            # belonging exclusively to ghost elements
            # and build edge connectivity device array
            edgconnoff = Int64[]
            temp = Int64[]
            push!(edgconnoff, 1)
            for i in 1:uedgno
                elen = length(edgconn[i])
                encls = Int(elen / 3)
                edgmrk = true
                if encls > 1
                    for j in 1:encls
                        if edgconn[i][3 * j] ≤ nreal
                            edgmrk = false
                        end
                    end
                end
                if edgmrk
                    edgconn[i] = []
                else
                    shift = edgconnoff[end]
                    push!(edgconnoff, (shift + length(edgconn[i])))
                    append!(temp, edgconn[i][:])
                end
            end
            edgconn = temp
        else
            edgconn = nothing
            edgconnoff = nothing
        end
    else
        elemtouvert = nothing
        vtconn = nothing
        vtconnoff = nothing
        fcconn = nothing
        edgconn = nothing
        edgconnoff = nothing
    end
    #---------------
    StackedBrickTopology{dim, T, nb}(
        BoxElementTopology{dim, T, nb}(
            mpicomm,
            elems,
            realelems,
            ghostelems,
            ghostfaces,
            sendelems,
            sendfaces,
            elemtocoord,
            elemtoelem,
            elemtoface,
            elemtoordr,
            elemtobndy,
            nabrtorank,
            nabrtorecv,
            nabrtosend,
            basetopo.origsendorder,
            bndytoelem,
            bndytoface,
            elemtouvert = elemtouvert,
            vtconn = vtconn,
            vtconnoff = vtconnoff,
            fcconn = fcconn,
            edgconn = edgconn,
            edgconnoff = edgconnoff,
        ),
        stacksize,
        periodicity[end],
    )
end

"""
    CubedShellTopology(mpicomm, Nelem, T) <: AbstractTopology{dim,T,nb}

Generate a cubed shell mesh with the number of elements along each dimension of
the cubes being `Nelem`. This topology actually creates a cube mesh, and the
warping should be done after the grid is created using the `cubed_sphere_warp`
function. The coordinates of the points will be of type `T`.

The elements of the shell are partitioned equally across the MPI ranks based
on a space-filling curve.

Note that this topology is logically 2-D but embedded in a 3-D space

# Examples

We can build a cubed shell mesh with 10*10 elements on each cube face, i.e., the
total number of elements is
`10 * 10 * 6 = 600`, with
```jldoctest brickmesh
using ClimateMachine.Topologies
using MPI
MPI.Init()
topology = CubedShellTopology(MPI.COMM_SELF, 10, Float64)

# Typically the warping would be done after the grid is created, but the cell
# corners could be warped with...

# Shell radius = 1
x1, x2, x3 = ntuple(j->topology.elemtocoord[j, :, :], 3)
for n = 1:length(x1)
   x1[n], x2[n], x3[n] = Topologies.cubed_sphere_warp(EquiangularCubedSphere(), x1[n], x2[n], x3[n])
end

# in case a unitary equiangular cubed sphere is desired, or

# Shell radius = 10
x1, x2, x3 = ntuple(j->topology.elemtocoord[j, :, :], 3)
for n = 1:length(x1)
  x1[n], x2[n], x3[n] = Topologies.cubed_sphere_warp(EquidistantCubedSphere(), x1[n], x2[n], x3[n], 10)
end

# in case an equidistant cubed sphere of radius 10 is desired.
```
"""
function CubedShellTopology(
    mpicomm,
    Neside,
    T;
    connectivity = :full,
    ghostsize = 1,
)

    # We cannot handle anything else right now...
    @assert ghostsize == 1

    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)

    topology = cubedshellmesh(Neside, part = mpirank + 1, numparts = mpisize)

    topology = BrickMesh.partition(mpicomm, topology...)
    origsendorder = topology[5]
    dim, nvert = 3, 4
    elemtovert = topology[1]
    nelem = size(elemtovert, 2)
    elemtocoord = Array{T}(undef, dim, nvert, nelem)
    ind2vert = CartesianIndices((Neside + 1, Neside + 1, Neside + 1))
    for e in 1:nelem
        for n in 1:nvert
            v = elemtovert[n, e]
            i, j, k = Tuple(ind2vert[v])
            elemtocoord[:, n, e] =
                (2 * [i - 1, j - 1, k - 1] .- Neside) / Neside
        end
    end

    topology =
        connectivity == :face ?
        BrickMesh.connectmesh(
            mpicomm,
            topology[1],
            elemtocoord,
            topology[3],
            topology[4],
            dim = 2,
        ) :
        BrickMesh.connectmeshfull(
            mpicomm,
            topology[1],
            elemtocoord,
            topology[3],
            topology[4],
            dim = 2,
        )

    CubedShellTopology{T}(BoxElementTopology{2, T, 0}(
        mpicomm,
        topology.elems,
        topology.realelems,
        topology.ghostelems,
        topology.ghostfaces,
        topology.sendelems,
        topology.sendfaces,
        topology.elemtocoord,
        topology.elemtoelem,
        topology.elemtoface,
        topology.elemtoordr,
        topology.elemtobndy,
        topology.nabrtorank,
        topology.nabrtorecv,
        topology.nabrtosend,
        origsendorder,
        (),
        (),
        elemtouvert = topology.elemtovert,
    ))
end

"""
    cubedshellmesh(T, Ne; part=1, numparts=1)

Generate a cubed mesh where each of the "cubes" has an `Ne X Ne` grid of
elements.

The mesh can optionally be partitioned into `numparts` and this returns
partition `part`. This is a simple Cartesian partition and further partitioning
(e.g, based on a space-filling curve) should be done before the mesh is used for
computation.

This mesh returns the cubed spehere in a flattened fashion for the vertex values,
and a remapping is needed to embed the mesh in a 3-D space.

The mesh structures for the cubes is as follows:

```
x2
   ^
   |
4Ne-           +-------+
   |           |       |
   |           |   6   |
   |           |       |
3Ne-           +-------+
   |           |       |
   |           |   5   |
   |           |       |
2Ne-           +-------+
   |           |       |
   |           |   4   |
   |           |       |
 Ne-   +-------+-------+-------+
   |   |       |       |       |
   |   |   1   |   2   |   3   |
   |   |       |       |       |
  0-   +-------+-------+-------+
   |
   +---|-------|-------|------|-> x1
       0      Ne      2Ne    3Ne
```

"""
function cubedshellmesh(Ne; part = 1, numparts = 1)
    dim = 2
    @assert 1 <= part <= numparts

    globalnelems = 6 * Ne^2

    # How many vertices and faces per element
    nvert = 2^dim # 4
    nface = 2dim  # 4

    # linearly partition to figure out which elements we own
    elemlocal = BrickMesh.linearpartition(prod(globalnelems), part, numparts)

    # elemen to vertex maps which we own
    elemtovert = Array{Int}(undef, nvert, length(elemlocal))
    elemtocoord = Array{Int}(undef, dim, nvert, length(elemlocal))

    nelemcube = Ne^dim # Ne^2

    etoijb = CartesianIndices((Ne, Ne, 6))
    bx = [0 Ne 2Ne Ne Ne Ne]
    by = [0 0 0 Ne 2Ne 3Ne]

    vertmap = LinearIndices((Ne + 1, Ne + 1, Ne + 1))
    for (le, e) in enumerate(elemlocal)
        i, j, blck = Tuple(etoijb[e])
        elemtocoord[1, :, le] = bx[blck] .+ [i - 1 i i - 1 i]
        elemtocoord[2, :, le] = by[blck] .+ [j - 1 j - 1 j j]

        for n in 1:4
            ix = i + mod(n - 1, 2)
            jx = j + div(n - 1, 2)
            # set the vertices like they are the face vertices of a cube
            if blck == 1
                elemtovert[n, le] = vertmap[1, Ne + 2 - ix, jx]
            elseif blck == 2
                elemtovert[n, le] = vertmap[ix, 1, jx]
            elseif blck == 3
                elemtovert[n, le] = vertmap[Ne + 1, ix, jx]
            elseif blck == 4
                elemtovert[n, le] = vertmap[ix, jx, Ne + 1]
            elseif blck == 5
                elemtovert[n, le] = vertmap[ix, Ne + 1, Ne + 2 - jx]
            elseif blck == 6
                elemtovert[n, le] = vertmap[ix, Ne + 2 - jx, 1]
            end
        end
    end

    # no boundaries for a shell
    elemtobndy = zeros(Int, nface, length(elemlocal))

    # no faceconnections for a shell
    faceconnections = Array{Array{Int, 1}}(undef, 0)

    (elemtovert, elemtocoord, elemtobndy, faceconnections, collect(elemlocal))
end

abstract type AbstractCubedSphere end
struct EquiangularCubedSphere <: AbstractCubedSphere end
struct EquidistantCubedSphere <: AbstractCubedSphere end
struct ConformalCubedSphere <: AbstractCubedSphere end

"""
    cubed_sphere_warp(::EquiangularCubedSphere, a, b, c, R = max(abs(a), abs(b), abs(c)))

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equiangular gnomonic grid proposed by
[Ronchi1996](@cite)
"""
function cubed_sphere_warp(
    ::EquiangularCubedSphere,
    a,
    b,
    c,
    R = max(abs(a), abs(b), abs(c)),
)

    function f(sR, ξ, η)
        X, Y = tan(π * ξ / 4), tan(π * η / 4)
        ζ1 = sR / sqrt(X^2 + Y^2 + 1)
        ζ2, ζ3 = X * ζ1, Y * ζ1
        ζ1, ζ2, ζ3
    end

    fdim = argmax(abs.((a, b, c)))
    if fdim == 1 && a < 0
        # (-R, *, *) : formulas for Face I from Ronchi, Iacono, Paolucci (1996)
        #              but for us face II of the developed net of the cube
        x1, x2, x3 = f(-R, b / a, c / a)
    elseif fdim == 2 && b < 0
        # ( *,-R, *) : formulas for Face II from Ronchi, Iacono, Paolucci (1996)
        #              but for us face III of the developed net of the cube
        x2, x1, x3 = f(-R, a / b, c / b)
    elseif fdim == 1 && a > 0
        # ( R, *, *) : formulas for Face III from Ronchi, Iacono, Paolucci (1996)
        #              but for us face IV of the developed net of the cube
        x1, x2, x3 = f(R, b / a, c / a)
    elseif fdim == 2 && b > 0
        # ( *, R, *) : formulas for Face IV from Ronchi, Iacono, Paolucci (1996)
        #              but for us face I of the developed net of the cube
        x2, x1, x3 = f(R, a / b, c / b)
    elseif fdim == 3 && c > 0
        # ( *, *, R) : formulas for Face V from Ronchi, Iacono, Paolucci (1996)
        #              and the same for us on the developed net of the cube
        x3, x2, x1 = f(R, b / c, a / c)
    elseif fdim == 3 && c < 0
        # ( *, *,-R) : formulas for Face VI from Ronchi, Iacono, Paolucci (1996)
        #              and the same for us on the developed net of the cube
        x3, x2, x1 = f(-R, b / c, a / c)
    else
        error("invalid case for cubed_sphere_warp(::EquiangularCubedSphere): $a, $b, $c")
    end

    return x1, x2, x3
end

"""
    equiangular_cubed_sphere_warp(a, b, c, R = max(abs(a), abs(b), abs(c)))

A wrapper function for the cubed_sphere_warp function, when called with the
EquiangularCubedSphere type
"""
equiangular_cubed_sphere_warp(a, b, c, R = max(abs(a), abs(b), abs(c))) =
    cubed_sphere_warp(EquiangularCubedSphere(), a, b, c, R)

"""
    cubed_sphere_unwarp(x1, x2, x3)

The inverse of [`cubed_sphere_warp`](@ref). This function projects
a given point `(x_1, x_2, x_3)` from the surface of a sphere onto a cube
"""
function cubed_sphere_unwarp(::EquiangularCubedSphere, x1, x2, x3)

    function g(R, X, Y)
        ξ = atan(X) * 4 / pi
        η = atan(Y) * 4 / pi
        R, R * ξ, R * η
    end

    R = hypot(x1, x2, x3)
    fdim = argmax(abs.((x1, x2, x3)))

    if fdim == 1 && x1 < 0
        # (-R, *, *) : formulas for Face I from Ronchi, Iacono, Paolucci (1996)
        #              but for us face II of the developed net of the cube
        a, b, c = g(-R, x2 / x1, x3 / x1)
    elseif fdim == 2 && x2 < 0
        # ( *,-R, *) : formulas for Face II from Ronchi, Iacono, Paolucci (1996)
        #              but for us face III of the developed net of the cube
        b, a, c = g(-R, x1 / x2, x3 / x2)
    elseif fdim == 1 && x1 > 0
        # ( R, *, *) : formulas for Face III from Ronchi, Iacono, Paolucci (1996)
        #              but for us face IV of the developed net of the cube
        a, b, c = g(R, x2 / x1, x3 / x1)
    elseif fdim == 2 && x2 > 0
        # ( *, R, *) : formulas for Face IV from Ronchi, Iacono, Paolucci (1996)
        #              but for us face I of the developed net of the cube
        b, a, c = g(R, x1 / x2, x3 / x2)
    elseif fdim == 3 && x3 > 0
        # ( *, *, R) : formulas for Face V from Ronchi, Iacono, Paolucci (1996)
        #              and the same for us on the developed net of the cube
        c, b, a = g(R, x2 / x3, x1 / x3)
    elseif fdim == 3 && x3 < 0
        # ( *, *,-R) : formulas for Face VI from Ronchi, Iacono, Paolucci (1996)
        #              and the same for us on the developed net of the cube
        c, b, a = g(-R, x2 / x3, x1 / x3)
    else
        error("invalid case for cubed_sphere_unwarp(::EquiangularCubedSphere): $a, $b, $c")
    end

    return a, b, c
end

"""
    equiangular_cubed_sphere_unwarp(x1, x2, x3)

A wrapper function for the cubed_sphere_unwarp function, when called with the
EquiangularCubedSphere type
"""
equiangular_cubed_sphere_unwarp(x1, x2, x3) =
    cubed_sphere_unwarp(EquiangularCubedSphere(), x1, x2, x3)

"""
    cubed_sphere_warp(a, b, c, R = max(abs(a), abs(b), abs(c)))

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equidistant gnomonic grid outlined in
[Rancic1996](@cite) and [Nair2005](@cite)
"""
function cubed_sphere_warp(
    ::EquidistantCubedSphere,
    a,
    b,
    c,
    R = max(abs(a), abs(b), abs(c)),
)

    r = hypot(a, b, c)

    x1 = R * a / r
    x2 = R * b / r
    x3 = R * c / r

    return x1, x2, x3
end

"""
    equidistant_cubed_sphere_warp(a, b, c, R = max(abs(a), abs(b), abs(c)))

A wrapper function for the cubed_sphere_warp function, when called with the
EquidistantCubedSphere type
"""
equidistant_cubed_sphere_warp(a, b, c, R = max(abs(a), abs(b), abs(c))) =
    cubed_sphere_warp(EquidistantCubedSphere(), a, b, c, R)

"""
    cubed_sphere_unwarp(x1, x2, x3)

The inverse of [`cubed_sphere_warp`](@ref). This function projects
a given point `(x_1, x_2, x_3)` from the surface of a sphere onto a cube
"""
function cubed_sphere_unwarp(::EquidistantCubedSphere, x1, x2, x3)

    r = hypot(1, x2 / x1, x3 / x1)
    R = hypot(x1, x2, x3)

    a = r * x1
    b = r * x2
    c = r * x3

    m = max(abs(a), abs(b), abs(c))

    return a * R / m, b * R / m, c * R / m
end

"""
    equidistant_cubed_sphere_unwarp(x1, x2, x3)

A wrapper function for the cubed_sphere_unwarp function, when called with the
EquidistantCubedSphere type
"""
equidistant_cubed_sphere_unwarp(x1, x2, x3) =
    cubed_sphere_unwarp(EquidistantCubedSphere(), x1, x2, x3)

"""
    cubed_sphere_warp(::ConformalCubedSphere, a, b, c, R = max(abs(a), abs(b), abs(c)))

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the conformal grid proposed by
[Rancic1996](@cite)
"""
function cubed_sphere_warp(
    ::ConformalCubedSphere,
    a,
    b,
    c,
    R = max(abs(a), abs(b), abs(c)),
)

    fdim = argmax(abs.((a, b, c)))
    M = max(abs.((a, b, c))...)
    if fdim == 1 && a < 0
        # left face
        x1, x2, x3 = conformal_cubed_sphere_mapping(-b / M, c / M)
        x1, x2, x3 = RotX(π / 2) * RotY(-π / 2) * [x1, x2, x3]
    elseif fdim == 2 && b < 0
        # front face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, c / M)
        x1, x2, x3 = RotX(π / 2) * [x1, x2, x3]
    elseif fdim == 1 && a > 0
        # right face
        x1, x2, x3 = conformal_cubed_sphere_mapping(b / M, c / M)
        x1, x2, x3 = RotX(π / 2) * RotY(π / 2) * [x1, x2, x3]
    elseif fdim == 2 && b > 0
        # back face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, -c / M)
        x1, x2, x3 = RotX(-π / 2) * [x1, x2, x3]
    elseif fdim == 3 && c > 0
        # top face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, b / M)
    elseif fdim == 3 && c < 0
        # bottom face
        x1, x2, x3 = conformal_cubed_sphere_mapping(a / M, -b / M)
        x1, x2, x3 = RotX(π) * [x1, x2, x3]
    else
        error("invalid case for cubed_sphere_warp(::ConformalCubedSphere): $a, $b, $c")
    end

    return x1 * R, x2 * R, x3 * R
end

"""
    conformal_cubed_sphere_warp(a, b, c, R = max(abs(a), abs(b), abs(c)))

A wrapper function for the cubed_sphere_warp function, when called with the
ConformalCubedSphere type
"""
conformal_cubed_sphere_warp(a, b, c, R = max(abs(a), abs(b), abs(c))) =
    cubed_sphere_warp(ConformalCubedSphere(), a, b, c, R)

"""
   StackedCubedSphereTopology(mpicomm, Nhorz, Rrange;
                              boundary=(1,1)) <: AbstractTopology{3}

Generate a stacked cubed sphere topology with `Nhorz` by `Nhorz` cells for each
horizontal face and `Rrange` is the radius edges of the stacked elements. This
topology actual creates a cube mesh, and the warping should be done after the
grid is created using the `cubed_sphere_warp` function. The coordinates of the
points will be of type `eltype(Rrange)`. The inner boundary condition type is
`boundary[1]` and the outer boundary condition type is `boundary[2]`.

The elements are stacked such that the vertical elements are contiguous in the
element ordering.

The elements of the brick are partitioned equally across the MPI ranks based
on a space-filling curve. Further, stacks are not split at MPI boundaries.

# Examples

We can build a cubed sphere mesh with 10 x 10 x 5 elements on each cube face,
i.e., the total number of elements is `10 * 10 * 5 * 6 = 3000`, with
```jldoctest brickmesh
using ClimateMachine.Topologies
using MPI
MPI.Init()
Nhorz = 10
Nstack = 5
Rrange = Float64.(accumulate(+,1:Nstack+1))
topology = StackedCubedSphereTopology(MPI.COMM_SELF, Nhorz, Rrange)

x1, x2, x3 = ntuple(j->reshape(topology.elemtocoord[j, :, :],
                            2, 2, 2, length(topology.elems)), 3)
for n = 1:length(x1)
   x1[n], x2[n], x3[n] = Topologies.cubed_sphere_warp(EquiangularCubedSphere(),x1[n], x2[n], x3[n])
end
```
Note that the faces are listed in Cartesian order.
"""
function StackedCubedSphereTopology(
    mpicomm,
    Nhorz,
    Rrange;
    boundary = (1, 1),
    connectivity = :full,
    ghostsize = 1,
)
    T = eltype(Rrange)

    basetopo = CubedShellTopology(
        mpicomm,
        Nhorz,
        T;
        connectivity = connectivity,
        ghostsize = ghostsize,
    )

    dim = 3
    nvert = 2^dim
    nface = 2dim
    stacksize = length(Rrange) - 1

    nreal = length(basetopo.realelems) * stacksize
    nghost = length(basetopo.ghostelems) * stacksize

    elems = 1:(nreal + nghost)
    realelems = 1:nreal
    ghostelems = nreal .+ (1:nghost)

    sendelems =
        similar(basetopo.sendelems, length(basetopo.sendelems) * stacksize)

    for i in 1:length(basetopo.sendelems), j in 1:stacksize
        sendelems[stacksize * (i - 1) + j] =
            stacksize * (basetopo.sendelems[i] - 1) + j
    end

    ghostfaces = similar(basetopo.ghostfaces, nface, length(ghostelems))
    ghostfaces .= false

    for i in 1:length(basetopo.ghostelems), j in 1:stacksize
        e = stacksize * (i - 1) + j
        for f in 1:(2 * (dim - 1))
            ghostfaces[f, e] = basetopo.ghostfaces[f, i]
        end
    end

    sendfaces = similar(basetopo.sendfaces, nface, length(sendelems))
    sendfaces .= false

    for i in 1:length(basetopo.sendelems), j in 1:stacksize
        e = stacksize * (i - 1) + j
        for f in 1:(2 * (dim - 1))
            sendfaces[f, e] = basetopo.sendfaces[f, i]
        end
    end

    elemtocoord = similar(basetopo.elemtocoord, dim, nvert, length(elems))

    for i in 1:length(basetopo.elems), j in 1:stacksize
        # i is base element
        # e is stacked element
        e = stacksize * (i - 1) + j


        # v is base vertex
        for v in 1:(2^(dim - 1))
            for d in 1:dim # dim here since shell is embedded in 3-D
                # v is lower stacked vertex
                elemtocoord[d, v, e] = basetopo.elemtocoord[d, v, i] * Rrange[j]
                # 2^(dim-1) + v is higher stacked vertex
                elemtocoord[d, 2^(dim - 1) + v, e] =
                    basetopo.elemtocoord[d, v, i] * Rrange[j + 1]
            end
        end
    end

    elemtoelem = similar(basetopo.elemtoelem, nface, length(elems))
    elemtoface = similar(basetopo.elemtoface, nface, length(elems))
    elemtoordr = similar(basetopo.elemtoordr, nface, length(elems))
    elemtobndy = similar(basetopo.elemtobndy, nface, length(elems))

    for e in 1:(length(basetopo.elems) * stacksize), f in 1:nface
        elemtoelem[f, e] = e
        elemtoface[f, e] = f
        elemtoordr[f, e] = 1
        elemtobndy[f, e] = 0
    end

    for i in 1:length(basetopo.realelems), j in 1:stacksize
        e1 = stacksize * (i - 1) + j

        for f in 1:(2 * (dim - 1))
            e2 = stacksize * (basetopo.elemtoelem[f, i] - 1) + j

            elemtoelem[f, e1] = e2
            elemtoface[f, e1] = basetopo.elemtoface[f, i]

            # since the basetopo is 2-D we only need to worry about two
            # orientations
            @assert basetopo.elemtoordr[f, i] ∈ (1, 2)
            #=
            orientation 1:
            2---3     2---3
            |   | --> |   |
            0---1     0---1
            same:
            (a,b) --> (a,b)

            orientation 3:
            2---3     3---2
            |   | --> |   |
            0---1     1---0
            reverse first index:
            (a,b) --> (N+1-a,b)
            =#
            elemtoordr[f, e1] = basetopo.elemtoordr[f, i] == 1 ? 1 : 3
        end

        # If top or bottom of stack set neighbor to self on respective face
        elemtoelem[2 * (dim - 1) + 1, e1] =
            j == 1 ? e1 : stacksize * (i - 1) + j - 1
        elemtoelem[2 * (dim - 1) + 2, e1] =
            j == stacksize ? e1 : stacksize * (i - 1) + j + 1

        elemtoface[2 * (dim - 1) + 1, e1] =
            j == 1 ? 2 * (dim - 1) + 1 : 2 * (dim - 1) + 2
        elemtoface[2 * (dim - 1) + 2, e1] =
            j == stacksize ? 2 * (dim - 1) + 2 : 2 * (dim - 1) + 1

        elemtoordr[2 * (dim - 1) + 1, e1] = 1
        elemtoordr[2 * (dim - 1) + 2, e1] = 1
    end

    # Set the top and bottom boundary condition
    for i in 1:length(basetopo.elems)
        eb = stacksize * (i - 1) + 1
        et = stacksize * (i - 1) + stacksize

        elemtobndy[2 * (dim - 1) + 1, eb] = boundary[1]
        elemtobndy[2 * (dim - 1) + 2, et] = boundary[2]
    end

    nabrtorank = basetopo.nabrtorank
    nabrtorecv = UnitRange{Int}[
        UnitRange(
            stacksize * (first(basetopo.nabrtorecv[n]) - 1) + 1,
            stacksize * last(basetopo.nabrtorecv[n]),
        ) for n in 1:length(nabrtorank)
    ]
    nabrtosend = UnitRange{Int}[
        UnitRange(
            stacksize * (first(basetopo.nabrtosend[n]) - 1) + 1,
            stacksize * last(basetopo.nabrtosend[n]),
        ) for n in 1:length(nabrtorank)
    ]

    bndytoelem, bndytoface = BrickMesh.enumerateboundaryfaces!(
        elemtoelem,
        elemtobndy,
        (false,),
        (boundary,),
    )
    nb = length(bndytoelem)
    #----setting up DSS----------
    if basetopo.elemtouvert isa Array
        #---setup vertex DSS
        nvertby2 = div(nvert, 2)
        elemtouvert = similar(basetopo.elemtouvert, nvert, length(elems))
        # base level
        for i in 1:length(basetopo.elems)
            e = stacksize * (i - 1) + 1
            for vt in 1:nvertby2
                elemtouvert[vt, e] =
                    (basetopo.elemtouvert[vt, i] - 1) * (stacksize + 1) + 1
            end
        end
        # interior levels
        for i in 1:length(basetopo.elems), j in 1:(stacksize - 1)
            e = stacksize * (i - 1) + j
            for vt in 1:nvertby2
                elemtouvert[nvertby2 + vt, e] = elemtouvert[vt, e] + 1
                elemtouvert[vt, e + 1] = elemtouvert[nvertby2 + vt, e]
            end
        end
        # top level
        for i in 1:length(basetopo.elems)
            e = stacksize * i
            for vt in 1:nvertby2
                elemtouvert[nvertby2 + vt, e] = elemtouvert[vt, e] + 1
            end
        end
        vtmax = maximum(unique(elemtouvert))
        vtconn = map(j -> zeros(Int, j), zeros(Int, vtmax))

        for el in elems, lvt in 1:nvert
            vt = elemtouvert[lvt, el]
            push!(vtconn[vt], lvt) # local vertex number
            push!(vtconn[vt], el)  # local elem number
        end
        # building the vertex connectivity device array
        vtconnoff = zeros(Int, vtmax + 1)
        vtconnoff[1] = 1
        temp = Int64[]
        for vt in 1:vtmax
            nconn = length(vtconn[vt])
            if nconn > 2
                vtconnoff[vt + 1] = vtconnoff[vt] + nconn
                append!(temp, vtconn[vt])
            else
                vtconnoff[vt + 1] = vtconnoff[vt]
            end
        end
        vtconn = temp
        #---setup face DSS---------------------
        fcmarker = -ones(Int, nface, length(elems))
        fcno = 0
        for el in realelems
            for fc in 1:nface
                if elemtobndy[fc, el] == 0
                    nabrel = elemtoelem[fc, el]
                    nabrfc = elemtoface[fc, el]
                    if fcmarker[fc, el] == -1 && fcmarker[nabrfc, nabrel] == -1
                        fcno += 1
                        fcmarker[fc, el] = fcno
                        fcmarker[nabrfc, nabrel] = fcno
                    end
                end
            end
        end
        fcconn = -ones(Int, fcno, 5)
        for el in realelems
            for fc in 1:nface
                fcno = fcmarker[fc, el]
                ordr = elemtoordr[fc, el]
                if fcno ≠ -1
                    nabrel = elemtoelem[fc, el]
                    nabrfc = elemtoface[fc, el]
                    fcconn[fcno, 1] = fc
                    fcconn[fcno, 2] = el
                    fcconn[fcno, 3] = nabrfc
                    fcconn[fcno, 4] = nabrel
                    fcconn[fcno, 5] = ordr
                    fcmarker[fc, el] = -1
                    fcmarker[nabrfc, nabrel] = -1
                end
            end
        end
        #---setup edge DSS---------------------
        if dim == 3
            nedge = 12
            edgemask = [
                1 3 5 7 1 2 5 6 1 2 3 4
                2 4 6 8 3 4 7 8 5 6 7 8
            ]
            edges = Array{Int64}(undef, 6, nedge * length(elems))
            ledge = zeros(Int64, 2)
            uedge = Dict{Array{Int64, 1}, Int64}()
            orient = 1
            uedgno = Int64(0)
            ctr = 1
            for el in elems
                for edg in 1:nedge
                    orient = 1
                    for i in 1:2
                        ledge[i] = elemtouvert[edgemask[i, edg], el]
                        edges[i, ctr] = ledge[i]                     # edge vertices [1:2]
                    end
                    if ledge[1] > ledge[2]
                        sort!(ledge)
                        orient = 2
                    end
                    edges[3, ctr] = edg    # local edge number
                    edges[4, ctr] = orient # edge orientation
                    edges[5, ctr] = el     # element number
                    if haskey(uedge, ledge[:])
                        edges[6, ctr] = uedge[ledge[:]]  # unique edge number
                    else
                        uedgno += 1
                        uedge[ledge[:]] = uedgno
                        edges[6, ctr] = uedgno
                    end
                    ctr += 1
                end
            end
            edgconn = map(j -> zeros(Int, j), zeros(Int, uedgno))
            ctr = 1
            for el in elems
                for edg in 1:nedge
                    uedg = edges[6, ctr]
                    push!(edgconn[uedg], edg)           # local edge number
                    push!(edgconn[uedg], edges[4, ctr]) # orientation
                    push!(edgconn[uedg], el)            # element number
                    ctr += 1
                end
            end
            # remove edges belonging to a single element and edges
            # belonging exclusively to ghost elements
            # and build edge connectivity device array
            edgconnoff = Int64[]
            temp = Int64[]
            push!(edgconnoff, 1)
            for i in 1:uedgno
                elen = length(edgconn[i])
                encls = Int(elen / 3)
                edgmrk = true
                if encls > 1
                    for j in 1:encls
                        if edgconn[i][3 * j] ≤ nreal
                            edgmrk = false
                        end
                    end
                end
                if edgmrk
                    edgconn[i] = []
                else
                    shift = edgconnoff[end]
                    push!(edgconnoff, (shift + length(edgconn[i])))
                    append!(temp, edgconn[i][:])
                end
            end
            edgconn = temp
        else
            edgconn = nothing
            edgconnoff = nothing
        end
        #-------------------------------------
    else
        elemtouvert = nothing
        vtconn = nothing
        vtconnoff = nothing
        fcconn = nothing
        edgconn = nothing
        edgconnoff = nothing
    end
    #-----------------------------
    StackedCubedSphereTopology{T, nb}(
        BoxElementTopology{3, T, nb}(
            mpicomm,
            elems,
            realelems,
            ghostelems,
            ghostfaces,
            sendelems,
            sendfaces,
            elemtocoord,
            elemtoelem,
            elemtoface,
            elemtoordr,
            elemtobndy,
            nabrtorank,
            nabrtorecv,
            nabrtosend,
            basetopo.origsendorder,
            bndytoelem,
            bndytoface,
            elemtouvert = elemtouvert,
            vtconn = vtconn,
            vtconnoff = vtconnoff,
            fcconn = fcconn,
            edgconn = edgconn,
            edgconnoff = edgconnoff,
        ),
        stacksize,
    )
end

"""
    grid1d(a, b[, stretch::AbstractGridStretching]; elemsize, nelem)

Discretize the 1D interval [`a`,`b`] into elements.
Exactly one of the following keyword arguments must be provided:
- `elemsize`: the average element size, or
- `nelem`: the number of elements.

The optional `stretch` argument allows stretching, otherwise the element sizes
will be uniform.

Returns either a range object or a vector containing the element boundaries.
"""
function grid1d(a, b, stretch = nothing; elemsize = nothing, nelem = nothing)
    xor(nelem === nothing, elemsize === nothing) ||
        error("Either `elemsize` or `nelem` arguments must be provided")
    if elemsize !== nothing
        nelem = round(Int, abs(b - a) / elemsize)
    end
    grid1d(a, b, stretch, nelem)
end
function grid1d(a, b, ::Nothing, nelem)
    range(a, stop = b, length = nelem + 1)
end

# TODO: document these
abstract type AbstractGridStretching end

"""
    SingleExponentialStretching(A)

Apply single-exponential stretching: `A > 0` will increase the density of points
at the lower boundary, `A < 0` will increase the density at the upper boundary.

# Reference
* "Handbook of Grid Generation" J. F. Thompson, B. K. Soni, N. P. Weatherill
  (Editors) RCR Press 1999, §3.6.1 Single-Exponential Function
"""
struct SingleExponentialStretching{T} <: AbstractGridStretching
    A::T
end
function grid1d(
    a::A,
    b::B,
    stretch::SingleExponentialStretching,
    nelem,
) where {A, B}
    F = float(promote_type(A, B))
    s = range(zero(F), stop = one(F), length = nelem + 1)
    a .+ (b - a) .* expm1.(stretch.A .* s) ./ expm1(stretch.A)
end

struct InteriorStretching{T} <: AbstractGridStretching
    attractor::T
end
function grid1d(a::A, b::B, stretch::InteriorStretching, nelem) where {A, B}
    F = float(promote_type(A, B))
    coe = F(2.5)
    s = range(zero(F), stop = one(F), length = nelem + 1)
    range(a, stop = b, length = nelem + 1) .+
    coe .* (stretch.attractor .- (b - a) .* s) .* (1 .- s) .* s
end

function basic_topology_info(topology::AbstractStackedTopology)
    nelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)
    nrealelem = length(topology.realelems)
    nhorzrealelem = div(nrealelem, nvertelem)

    return (
        nelem = nelem,
        nvertelem = nvertelem,
        nhorzelem = nhorzelem,
        nrealelem = nrealelem,
        nhorzrealelem = nhorzrealelem,
    )
end

function basic_topology_info(topology::AbstractTopology)
    return (
        nelem = length(topology.elems),
        nrealelem = length(topology.realelems),
    )
end


### Helper Functions for Topography Calculations
"""
    compute_lat_long(X,Y,δ,faceid)
Helper function to allow computation of latitute and longitude coordinates
given the cubed sphere coordinates X, Y, δ, faceid
"""
function compute_lat_long(X, Y, δ, faceid)
    if faceid == 1
        λ = atan(X)                     # longitude
        ϕ = atan(cos(λ) * Y)            # latitude
    elseif faceid == 2
        λ = atan(X) + π / 2
        ϕ = atan(Y * cos(atan(X)))
    elseif faceid == 3
        λ = atan(X) + π
        ϕ = atan(Y * cos(atan(X)))
    elseif faceid == 4
        λ = atan(X) + (3 / 2) * π
        ϕ = atan(Y * cos(atan(X)))
    elseif faceid == 5
        λ = atan(X, -Y) + π
        ϕ = atan(1 / sqrt(δ - 1))
    elseif faceid == 6
        λ = atan(X, Y)
        ϕ = -atan(1 / sqrt(δ - 1))
    end
    return λ, ϕ
end


"""
    AnalyticalTopography
Abstract type to allow dispatch over different analytical topography prescriptions
in experiments.
"""
abstract type AnalyticalTopography end

function compute_analytical_topography(
    ::AnalyticalTopography,
    λ,
    ϕ,
    sR,
    r_inner,
    r_outer,
)
    return sR
end

"""
    NoTopography <: AnalyticalTopography
Allows definition of fallback methods in case cubed_sphere_topo_warp is used with
no prescribed topography function.
"""
struct NoTopography <: AnalyticalTopography end

### DCMIP Mountain
"""
    DCMIPMountain <: AnalyticalTopography
Topography description based on standard DCMIP experiments.
"""
struct DCMIPMountain <: AnalyticalTopography end
function compute_analytical_topography(
    ::DCMIPMountain,
    λ,
    ϕ,
    sR,
    r_inner,
    r_outer,
)
    #User specified warp parameters
    R_m = π * 3 / 4
    h0 = 2000
    ζ_m = π / 16
    φ_m = 0
    λ_m = π * 3 / 2
    r_m = acos(sin(φ_m) * sin(ϕ) + cos(φ_m) * cos(ϕ) * cos(λ - λ_m))
    # Define mesh decay profile
    Δ = (r_outer - abs(sR)) / (r_outer - r_inner)
    if r_m < R_m
        zs =
            0.5 *
            h0 *
            (1 + cos(π * r_m / R_m)) *
            cos(π * r_m / ζ_m) *
            cos(π * r_m / ζ_m)
    else
        zs = 0.0
    end
    mR = sign(sR) * (abs(sR) + zs * Δ)
    return mR
end

"""
    cubed_sphere_topo_warp(a, b, c, R = max(abs(a), abs(b), abs(c));
                       r_inner = _planet_radius,
                       r_outer = _planet_radius + domain_height,
                       topography = NoTopography())

Given points `(a, b, c)` on the surface of a cube, warp the points out to a
spherical shell of radius `R` based on the equiangular gnomonic grid proposed by
[Ronchi1996](@cite). Assumes a user specified modified radius using the
compute_analytical_topography function. Defaults to smooth cubed sphere unless otherwise specified
via the AnalyticalTopography type.
"""
function cubed_sphere_topo_warp(
    a,
    b,
    c,
    R = max(abs(a), abs(b), abs(c));
    r_inner = _planet_radius,
    r_outer = _planet_radius + domain_height,
    topography::AnalyticalTopography = NoTopography(),
)

    function f(sR, ξ, η, faceid)
        X, Y = tan(π * ξ / 4), tan(π * η / 4)
        δ = 1 + X^2 + Y^2
        λ, ϕ = compute_lat_long(X, Y, δ, faceid)
        mR = compute_analytical_topography(
            topography,
            λ,
            ϕ,
            sR,
            r_inner,
            r_outer,
        )
        x1 = mR / sqrt(δ)
        x2, x3 = X * x1, Y * x1
        x1, x2, x3
    end
    fdim = argmax(abs.((a, b, c)))

    if fdim == 1 && a < 0
        faceid = 1
        # (-R, *, *) : Face I from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = f(-R, b / a, c / a, faceid)
    elseif fdim == 2 && b < 0
        faceid = 2
        # ( *,-R, *) : Face II from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = f(-R, a / b, c / b, faceid)
    elseif fdim == 1 && a > 0
        faceid = 3
        # ( R, *, *) : Face III from Ronchi, Iacono, Paolucci (1996)
        x1, x2, x3 = f(R, b / a, c / a, faceid)
    elseif fdim == 2 && b > 0
        faceid = 4
        # ( *, R, *) : Face IV from Ronchi, Iacono, Paolucci (1996)
        x2, x1, x3 = f(R, a / b, c / b, faceid)
    elseif fdim == 3 && c > 0
        faceid = 5
        # ( *, *, R) : Face V from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = f(R, b / c, a / c, faceid)
    elseif fdim == 3 && c < 0
        faceid = 6
        # ( *, *,-R) : Face VI from Ronchi, Iacono, Paolucci (1996)
        x3, x2, x1 = f(-R, b / c, a / c, faceid)
    else
        error("invalid case for cubed_sphere_warp(::EquiangularCubedSphere): $a, $b, $c")
    end

    return x1, x2, x3
end

end
