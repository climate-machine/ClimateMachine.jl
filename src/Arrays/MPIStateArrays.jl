module MPIStateArrays

using CUDA
using DoubleFloats
using KernelAbstractions
using LazyArrays
using LinearAlgebra
using MPI
using StaticArrays

using ..TicToc
using ..VariableTemplates:
    @vars, varsindex, varsindices, varsize, wrap_val, flattened_tup_chain

using Base.Broadcast: Broadcasted, BroadcastStyle, ArrayStyle

struct ErrorOnRemoteNode <: Exception end

# This is so we can do things like
#   similar(Array{Float64}, Int, 3, 4)
Base.similar(::Type{A}, ::Type{FT}, dims...) where {A <: Array, FT} =
    similar(Array{FT}, dims...)

Base.similar(::Type{A}, ::Type{FT}, dims...) where {A <: CuArray, FT} =
    similar(CuArray{FT}, dims...)

include("CMBuffers.jl")
using .CMBuffers

cpuify(x::AbstractArray) = convert(Array, x)
cpuify(x::Real) = x

export MPIStateArray,
    euclidean_distance,
    weightedsum,
    array_device,
    vars,
    show_not_finite_fields,
    ErrorOnRemoteNode,
    checked_wait

"""
    MPIStateArray{FT, DATN<:AbstractArray{FT,3}, DAI1, DAV,
                  DAT2<:AbstractArray{FT,2}} <: AbstractArray{FT, 3}
"""
mutable struct MPIStateArray{
    FT,
    V,
    DATN <: AbstractArray{FT, 3},
    DAI1,
    DAV,
    Buf <: CMBuffer,
} <: AbstractArray{FT, 3}
    mpicomm::MPI.Comm
    data::DATN
    realdata::DAV

    realelems::UnitRange{Int64}
    ghostelems::UnitRange{Int64}

    vmaprecv::DAI1
    vmapsend::DAI1

    sendreq::Array{MPI.Request, 1}
    recvreq::Array{MPI.Request, 1}

    send_buffer::Buf
    recv_buffer::Buf

    nabrtorank::Array{Int64, 1}
    nabrtovmaprecv::Array{UnitRange{Int64}, 1}
    nabrtovmapsend::Array{UnitRange{Int64}, 1}

    weights::DATN

    function MPIStateArray{FT, V}(
        mpicomm,
        DA,
        Np,
        nstate,
        numelem,
        realelems,
        ghostelems,
        vmaprecv,
        vmapsend,
        nabrtorank,
        nabrtovmaprecv,
        nabrtovmapsend,
        weights;
        mpi_knows_cuda = nothing,
    ) where {FT, V}
        data = similar(DA, FT, Np, nstate, numelem)


        if varsize(V) > 0 && varsize(V) != nstate
            error("var sizes and numbers of states do not match")
        end

        if isnothing(mpi_knows_cuda)
            mpi_knows_cuda = MPI.has_cuda()
        end

        if data isa Array || mpi_knows_cuda
            kind = SingleCMBuffer
        else
            kind = DoubleCMBuffer
        end
        recv_buffer = CMBuffer{FT}(DA, kind, nstate, length(vmaprecv))
        send_buffer = CMBuffer{FT}(DA, kind, nstate, length(vmapsend))

        realdata =
            view(data, ntuple(i -> Colon(), ndims(data) - 1)..., realelems)
        DAV = typeof(realdata)

        nnabr = length(nabrtorank)
        sendreq = fill(MPI.REQUEST_NULL, nnabr)
        recvreq = fill(MPI.REQUEST_NULL, nnabr)

        # If vmap is not on the device we need to copy it up (we also do not want to
        # put it up everytime, so if it's already on the device then we do not do
        # anything).
        #
        # Better way than checking the type names?
        # XXX: Use Adapt.jl vmaprecv = adapt(DA, vmaprecv)
        if typeof(vmaprecv).name != typeof(data).name
            vmaprecv =
                copyto!(similar(DA, eltype(vmaprecv), size(vmaprecv)), vmaprecv)
        end
        if typeof(vmapsend).name != typeof(data).name
            vmapsend =
                copyto!(similar(DA, eltype(vmapsend), size(vmapsend)), vmapsend)
        end
        if typeof(weights).name != typeof(data).name
            weights = copyto!(
                similar(DA, eltype(weights), size(weights)),
                Array(weights),
            )
        end

        DAI1 = typeof(vmaprecv)
        Buf = typeof(send_buffer)
        Q = new{FT, V, typeof(data), DAI1, DAV, Buf}(
            # Make sure that each MPIStateArray has its own MPI context.  This
            # allows multiple MPIStateArrays to be communicating asynchronously
            # at the same time without having to explicitly manage tags.
            MPI.Comm_dup(mpicomm),
            data,
            realdata,
            realelems,
            ghostelems,
            vmaprecv,
            vmapsend,
            sendreq,
            recvreq,
            send_buffer,
            recv_buffer,
            nabrtorank,
            nabrtovmaprecv,
            nabrtovmapsend,
            weights,
        )
        # Make sure that we have finished all outstanding data for halo
        # exchanges before the MPIStateArray is finalize.
        finalizer(Q) do x
            if !MPI.Finalized()
                MPI.Waitall!(x.recvreq)
                MPI.Waitall!(x.sendreq)
            end
        end
        return Q
    end
end

function Base.fill!(Q::MPIStateArray, x)
    fill!(Q.data, x)
    return Q
end

vars(Q::MPIStateArray{FT, V}) where {FT, V} = V
function Base.getproperty(Q::MPIStateArray{FT, V}, sym::Symbol) where {FT, V}
    if sym ∈ V.names
        varrange = varsindex(V, sym)
        return view(realview(Q), :, varrange, :)
    else
        return getfield(Q, sym)
    end
end

"""
    getstateview(Q, fieldname)

Returns of real element view of the state `fieldname`. The MPI state array `Q`
must have been generated in a `Vars`-aware fashion. This is mainly to allow accessing fields which line inside of the nested `Vars` state.

The state specified by 'fieldname' can be a 'String', 'Symbol', or 'Expr'

# Examples
```julia-repl
julia> S = @vars(x::Float64, y::@vars(α::Float64, β::SVector{3, Float64}))
julia> Q = MPIStateArray{Float64, S}(MPI.COMM_WORLD, Array, 2, 5, 3)
julia> β = MPIStateArrays.getstateview(Q, "y.β")
julia> β = MPIStateArrays.getstateview(Q, :(y.β))
julia> x = MPIStateArrays.getstateview(Q, :x)
```
"""
function getstateview(
    Q::MPIStateArray{FT, V},
    fieldname::Union{String, Symbol, Expr},
) where {FT, V}
    varrange = varsindices(V, fieldname)
    return view(realview(Q), :, varrange[1]:varrange[end], :)
end

"""
   MPIStateArray{FT, V}(mpicomm, DA, Np, nstate, numelem; realelems=1:numelem,
                        ghostelems=numelem:numelem-1,
                        vmaprecv=1:0,
                        vmapsend=1:0,
                        nabrtorank=Array{Int64}(undef, 0),
                        nabrtovmaprecv=Array{UnitRange{Int64}}(undef, 0),
                        nabrtovmapsend=Array{UnitRange{Int64}}(undef, 0),
                        weights)

Construct an `MPIStateArray` over the communicator `mpicomm` with `numelem`
elements, using array type `DA` with element type `FT`. `V` is an optional type
to associate the `MPIStateArray` with a given `NamedTuple` used for `Vars` (e.g,
generated by macro `@vars`). The arrays that are held in this created
`MPIStateArray` will be of size `(Np, nstate, numelem)`.

The range `realelems` is the number of elements that this mpirank owns, whereas
the range `ghostelems` is the elements that are owned by other mpiranks.
Elements are stored as 'realelems` followed by `ghostelems`.

  * `vmaprecv` is an ordered array of elements to be received from neighboring
     mpiranks.  This is a vmap index into the MPIStateArray `A[:,*,:]`.
  * `vmapsend` is an ordered array of elements to be sent to neighboring
     mpiranks.  This is a vmap index into the MPIStateArray `A[:,*,:]`.
  * `nabrtorank` is the list of neighboring mpiranks
  * `nabrtovmaprecv` is an `Array` of `UnitRange` that give the ghost data to be
    received from neighboring mpiranks (indexes into `vmaprecv`)
  * `nabrtovmapsend` is an `Array` of `UnitRange` for which elements to send to
    which neighboring mpiranks indexing into the `vmapsend`
  * `weights` is an optional array which gives weight for each degree of freedom
    to be used when computing the 2-norm of the array
"""
MPIStateArray{FT}(args...; kwargs...) where {FT} =
    MPIStateArray{FT, @vars()}(args...; kwargs...)

function MPIStateArray{FT, V}(
    mpicomm,
    DA,
    Np,
    nstate,
    numelem;
    realelems = 1:numelem,
    ghostelems = numelem:(numelem - 1),
    vmaprecv = 1:0,
    vmapsend = 1:0,
    nabrtorank = Array{Int64}(undef, 0),
    nabrtovmaprecv = Array{UnitRange{Int64}}(undef, 0),
    nabrtovmapsend = Array{UnitRange{Int64}}(undef, 0),
    weights = nothing,
    mpi_knows_cuda = nothing,
) where {FT, V}

    if weights == nothing
        weights = similar(DA, FT, ntuple(j -> 0, 3))
    end
    MPIStateArray{FT, V}(
        mpicomm,
        DA,
        Np,
        nstate,
        numelem,
        realelems,
        ghostelems,
        vmaprecv,
        vmapsend,
        nabrtorank,
        nabrtovmaprecv,
        nabrtovmapsend,
        weights,
        mpi_knows_cuda = mpi_knows_cuda,
    )
end

# FIXME: should general cases be handled?
function Base.similar(
    Q::MPIStateArray,
    ::Type{A},
    ::Type{FT};
    vars::Type{V} = vars(Q),
    nstate = size(Q.data)[2],
) where {A <: AbstractArray, FT <: Number, V}
    MPIStateArray{FT, V}(
        Q.mpicomm,
        A,
        size(Q.data)[1],
        nstate,
        size(Q.data)[3],
        Q.realelems,
        Q.ghostelems,
        Q.vmaprecv,
        Q.vmapsend,
        Q.nabrtorank,
        Q.nabrtovmaprecv,
        Q.nabrtovmapsend,
        Q.weights,
    )
end
function Base.similar(
    Q::MPIStateArray{FT},
    ::Type{A};
    vars::Type{V} = vars(Q),
    nstate = size(Q.data)[2],
) where {A <: AbstractArray, FT <: Number, V}
    similar(Q, A, FT; vars = V, nstate = nstate)
end
function Base.similar(
    Q::MPIStateArray,
    ::Type{FT};
    vars::Type{V} = vars(Q),
    nstate = size(Q.data)[2],
) where {FT <: Number, V}
    similar(Q, typeof(Q.data), FT; vars = V, nstate = nstate)
end
function Base.similar(
    Q::MPIStateArray{FT};
    vars::Type{V} = vars(Q),
    nstate = size(Q.data)[2],
) where {FT, V}
    similar(Q, FT; vars = V, nstate = nstate)
end

Base.size(Q::MPIStateArray, x...; kw...) = size(Q.realdata, x...; kw...)

Base.getindex(Q::MPIStateArray, x...; kw...) = getindex(Q.realdata, x...; kw...)

Base.setindex!(Q::MPIStateArray, x...; kw...) =
    setindex!(Q.realdata, x...; kw...)

Base.eltype(Q::MPIStateArray, x...; kw...) = eltype(Q.data, x...; kw...)

Base.Array(Q::MPIStateArray) = Array(Q.data)

# broadcasting stuff

# find the first MPIStateArray among `bc` arguments
# based on https://docs.julialang.org/en/v1/manual/interfaces/#Selecting-an-appropriate-output-array-1
find_mpisa(bc::Broadcasted) = find_mpisa(bc.args)
find_mpisa(args::Tuple) = find_mpisa(find_mpisa(args[1]), Base.tail(args))
find_mpisa(x) = x
find_mpisa(a::MPIStateArray, rest) = a
find_mpisa(::Any, rest) = find_mpisa(rest)

Base.BroadcastStyle(::Type{<:MPIStateArray}) = ArrayStyle{MPIStateArray}()
function Base.similar(
    bc::Broadcasted{ArrayStyle{MPIStateArray}},
    ::Type{FT},
) where {FT}
    similar(find_mpisa(bc), FT)
end

# transform all arguments of `bc` from MPIStateArrays to Arrays
function transform_broadcasted(bc::Broadcasted, ::Array)
    transform_array(bc)
end
function transform_array(bc::Broadcasted)
    Broadcasted(bc.f, transform_array.(bc.args), bc.axes)
end
transform_array(mpisa::MPIStateArray) = mpisa.realdata
transform_array(x) = x

Base.copyto!(dest::Array, src::MPIStateArray) = copyto!(dest, src.data)

function Base.copyto!(dest::MPIStateArray, src::MPIStateArray)
    copyto!(dest.realdata, src.realdata)
    dest
end

@inline function Base.copyto!(dest::MPIStateArray, bc::Broadcasted{Nothing})
    # check for the case a .= b, where b is an array
    if bc.f === identity && bc.args isa Tuple{AbstractArray}
        if bc.args isa Tuple{MPIStateArray}
            realindices = CartesianIndices((
                axes(dest.data)[1:(end - 1)]...,
                dest.realelems,
            ))
            copyto!(dest.data, realindices, bc.args[1].data, realindices)
        else
            copyto!(dest.data, bc.args[1])
        end
    else
        copyto!(dest.realdata, transform_broadcasted(bc, dest.data))
    end
    dest
end

@inline function Base.zero(Q::MPIStateArray{FT}) where {FT}
    S = similar(Q)
    fill!(S, 0)
    return S
end

"""
    begin_ghost_exchange!(Q::MPIStateArray; dependencies = nothing)

Begin the MPI halo exchange of the data stored in `Q`.  A KernelAbstractions
`Event` is returned that can be used as a dependency to end the exchange.
"""
function begin_ghost_exchange!(Q::MPIStateArray; dependencies = nothing)
    if !all(r -> r == MPI.REQUEST_NULL, Q.recvreq)
        error("The currrent ghost exchange must end before another begins.")
    end

    __Irecv!(Q)

    # wait on (prior) MPI sends
    @tic mpi_sendwait
    MPI.Waitall!(Q.sendreq)
    fill!(Q.sendreq, MPI.REQUEST_NULL)
    @toc mpi_sendwait

    @tic mpi_sendcopy
    stage = get_stage(Q.send_buffer)
    progress = () -> iprobe_and_yield(Q.mpicomm)
    event = fillsendbuf!(
        stage,
        Q.data,
        Q.vmapsend;
        dependencies = dependencies,
        progress = progress,
    )
    event = prepare_transfer!(
        Q.send_buffer;
        dependencies = event,
        progress = progress,
    )
    @toc mpi_sendcopy

    return event
end

"""
    end_ghost_exchange!(Q::MPIStateArray; dependencies = nothing)

This function blocks on the host until the ghost halo is received from MPI.  A
KernelAbstractions `Event` is returned that can be waited on to indicate when
the data is ready on the device.
"""
function end_ghost_exchange!(
    Q::MPIStateArray;
    dependencies = nothing,
    check_for_crashes = false,
)
    if any(r -> r == MPI.REQUEST_NULL, Q.recvreq)
        error("A ghost exchange must begin before it ends.")
    end

    progress = () -> iprobe_and_yield(Q.mpicomm)
    checked_wait(CPU(), MultiEvent(dependencies), progress, check_for_crashes)

    __Isend!(Q)

    @tic mpi_recvwait
    MPI.Waitall!(Q.recvreq)
    fill!(Q.recvreq, MPI.REQUEST_NULL)
    @toc mpi_recvwait

    @tic mpi_recvcopy
    event = prepare_stage!(Q.recv_buffer; progress = progress)
    stage = get_stage(Q.recv_buffer)
    event = transferrecvbuf!(
        Q.data,
        stage,
        Q.vmaprecv;
        dependencies = event,
        progress = progress,
    )
    @toc mpi_recvcopy

    return event
end

function __Irecv!(Q)
    nnabr = length(Q.nabrtorank)
    transfer = get_transfer(Q.recv_buffer)

    for n in 1:nnabr
        # If this fails we haven't waited on previous recv!
        @assert Q.recvreq[n].buffer == nothing

        Q.recvreq[n] = MPI.Irecv!(
            (@view transfer[:, Q.nabrtovmaprecv[n]]),
            Q.nabrtorank[n],
            666,
            Q.mpicomm,
        )
    end
end

function __Isend!(Q)
    nnabr = length(Q.nabrtorank)
    transfer = get_transfer(Q.send_buffer)

    for n in 1:nnabr
        Q.sendreq[n] = MPI.Isend(
            (@view transfer[:, Q.nabrtovmapsend[n]]),
            Q.nabrtorank[n],
            666,
            Q.mpicomm,
        )
    end
end

function iprobe_and_yield(comm)
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, comm)
    yield()
end

# {{{ MPI Buffer handling
function fillsendbuf!(
    sendbuf,
    buf,
    vmapsend;
    dependencies = nothing,
    progress = yield,
)
    if length(vmapsend) == 0
        return MultiEvent(dependencies)
    end

    Np = size(buf, 1)
    nvar = size(buf, 2)

    event = kernel_fillsendbuf!(array_device(buf), 256)(
        Val(Np),
        Val(nvar),
        sendbuf,
        buf,
        vmapsend,
        length(vmapsend);
        ndrange = length(vmapsend),
        dependencies = dependencies,
        progress = progress,
    )

    return event
end

function transferrecvbuf!(
    buf,
    recvbuf,
    vmaprecv;
    dependencies = nothing,
    progress = yield,
)
    if length(vmaprecv) == 0
        return MultiEvent(dependencies)
    end

    Np = size(buf, 1)
    nvar = size(buf, 2)

    event = kernel_transferrecvbuf!(array_device(buf), 256)(
        Val(Np),
        Val(nvar),
        buf,
        recvbuf,
        vmaprecv,
        length(vmaprecv);
        ndrange = length(vmaprecv),
        dependencies = dependencies,
        progress = progress,
    )

    return event
end

# }}}

# Integral based metrics
function LinearAlgebra.norm(
    Q::MPIStateArray,
    p::Real = 2,
    weighted::Bool = true;
    dims = :,
    realdata = Q.realdata,
)
    if weighted && ~isempty(Q.weights) && isfinite(p)
        W = @view Q.weights[:, :, Q.realelems]

        # Check that the arrays are the right dimensionality
        @assert ndims(realdata) == 3

        # Check that these are broadcastable
        @assert size(realdata, 1) == size(W, 1)
        @assert size(realdata, 3) == size(W, 3)

        locnorm = weighted_norm_impl(realdata, W, Val(p), dims)
    else
        locnorm = norm_impl(realdata, Val(p), dims)
    end

    mpiop = isfinite(p) ? (+) : max
    if locnorm isa AbstractArray
        locnorm = convert(Array, locnorm)
    end
    @tic mpi_norm
    r = MPI.Allreduce(cpuify(locnorm), mpiop, Q.mpicomm)
    @toc mpi_norm
    isfinite(p) ? r .^ (1 // p) : r
end
LinearAlgebra.norm(
    Q::MPIStateArray,
    weighted::Bool;
    dims = :,
    realdata = Q.realdata,
) = norm(Q, 2, weighted; dims = dims, realdata = realdata)

function LinearAlgebra.dot(
    Q1::MPIStateArray,
    Q2::MPIStateArray,
    weighted::Bool = true,
)
    @assert length(Q1.realdata) == length(Q2.realdata)

    if weighted && ~isempty(Q1.weights)
        W = @view Q1.weights[:, :, Q1.realelems]
        locnorm = weighted_dot_impl(Q1.realdata, Q2.realdata, W)
    else
        locnorm = dot_impl(Q1.realdata, Q2.realdata)
    end

    @tic mpi_dot
    r = MPI.Allreduce(locnorm, +, Q1.mpicomm)
    @toc mpi_dot
    return r
end

function euclidean_distance(
    A::MPIStateArray,
    B::MPIStateArray;
    ArealQ = A.realdata,
    BrealQ = B.realdata,
)
    # Check the the arrays are the right dimensionality
    @assert ndims(ArealQ) == 3
    @assert ndims(BrealQ) == 3

    # Check that these are broadcastable
    @assert size(ArealQ, 1) == size(BrealQ, 1)
    @assert size(ArealQ, 3) == size(BrealQ, 3)
    @assert size(ArealQ, 2) == 1 ||
            size(BrealQ, 2) == 1 ||
            size(ArealQ, 2) == size(BrealQ, 2)

    # work around https://github.com/JuliaArrays/LazyArrays.jl/issues/66
    E = @~ (ArealQ .- BrealQ) .^ 2

    if ~isempty(A.weights)
        w = @view A.weights[:, :, A.realelems]
        E = @~ E .* w
    end

    locnorm = mapreduce(identity, +, E, init = zero(eltype(A)))
    @tic mpi_euclidean_distance
    r = sqrt(MPI.Allreduce(locnorm, +, A.mpicomm))
    @toc mpi_euclidean_distance
    return r
end

"""
    weightedsum(A[, states])

Compute the weighted sum of the `MPIStateArray` `A`. If `states` is specified on
the listed states are summed, otherwise all the states in `A` are used.

A typical use case for this is when the weights have been initialized with
quadrature weights from a grid, thus this becomes an integral approximation.
"""
function weightedsum(A::MPIStateArray, states = 1:size(A, 2))
    isempty(A.weights) && error("`weightedsum` requires weights")

    FT = eltype(A)
    states = SVector{length(states)}(states)

    C = @view A.data[:, states, A.realelems]
    w = @view A.weights[:, :, A.realelems]
    init = zero(DoubleFloat{FT})

    E = @~ DoubleFloat{FT}.(C) .* DoubleFloat{FT}.(w)

    locwsum = mapreduce(identity, +, E, init = init)

    @tic mpi_weightedsum
    # Need to use anomous function version of sum else MPI.jl using MPI_SUM
    r = FT(MPI.Allreduce(locwsum, (x, y) -> x + y, A.mpicomm))
    @toc mpi_weightedsum
    return r
end

# fast CPU local norm & dot implementations
function norm_impl(
    Q::SubArray{FT, N, A},
    ::Val{p},
    dims::Colon,
) where {FT, N, A <: Array, p}
    accum = isfinite(p) ? -zero(FT) : typemin(FT)
    @inbounds @simd for i in eachindex(Q)
        if isfinite(p)
            accum += abs(Q[i])^p
        else
            aQ_i = abs(Q[i])
            accum = ifelse(aQ_i > accum, aQ_i, accum)
        end
    end
    accum
end

function weighted_norm_impl(
    Q::SubArray{FT, N, A},
    W,
    ::Val{p},
    dims::Colon,
) where {FT, N, A <: Array, p}
    @assert isfinite(p)
    nq, ns, ne = size(Q)
    accum = -zero(FT)
    @inbounds for k in 1:ne, j in 1:ns
        @simd for i in 1:nq
            accum += W[i, 1, k] * abs(Q[i, j, k])^p
        end
    end
    accum
end

function dot_impl(
    Q1::SubArray{FT, N, A},
    Q2::SubArray{FT, N, A},
) where {FT, N, A <: Array}
    accum = -zero(FT)
    @inbounds @simd for i in eachindex(Q1)
        accum += Q1[i] * Q2[i]
    end
    accum
end


function weighted_dot_impl(
    Q1::SubArray{FT, N, A},
    Q2::SubArray{FT, N, A},
    W,
) where {FT, N, A <: Array}
    nq, ns, ne = size(Q1)
    accum = -zero(FT)
    @inbounds for k in 1:ne, j in 1:ns
        @simd for i in 1:nq
            accum += W[i, 1, k] * Q1[i, j, k] * Q2[i, j, k]
        end
    end
    accum
end

# GPU/generic local norm & dot implementations
function norm_impl(Q, ::Val{p}, dims = :) where {p}
    FT = eltype(Q)
    if !isfinite(p)
        f, op = abs, max
    elseif p == 1
        f, op = abs, +
    elseif p == 2
        f, op = abs2, +
    else
        f, op = x -> abs(x)^p, +
    end
    mapreduce(f, op, Q, dims = dims)
end

function weighted_norm_impl(Q, W, ::Val{p}, dims = :) where {p}
    @assert isfinite(p)
    FT = eltype(Q)
    if p == 1
        E = @~ @. W * abs(Q)
    elseif p == 2
        E = @~ @. W * abs2(Q)
    else
        E = @~ @. W * abs(Q)^p
    end
    op, init = +, zero(FT)
    reduce(op, E, init = init, dims = dims)
end

function dot_impl(Q1, Q2)
    FT = eltype(Q1)
    E = @~ @. Q1 * Q2
    mapreduce(identity, +, E, init = zero(FT))
end

weighted_dot_impl(Q1, Q2, W) = dot_impl(@~ @. W * Q1, Q2)

function Base.mapreduce(f, op, Q::MPIStateArray; kw...)
    locreduce = mapreduce(f, op, realview(Q); kw...)
    MPI.Allreduce(cpuify(locreduce), op, Q.mpicomm)
end

# Arrays and CuArrays have different reduction machinery
# until we can figure this out, add special cases to make common functions work
function Base.mapreduce(
    ::typeof(identity),
    ::Union{typeof(+), typeof(Base.add_sum)},
    Q::MPIStateArray;
    kw...,
)
    locreduce = sum(realview(Q); kw...)
    MPI.Allreduce(cpuify(locreduce), +, Q.mpicomm)
end
function Base.mapreduce(
    ::typeof(identity),
    ::typeof(min),
    Q::MPIStateArray;
    kw...,
)
    locreduce = minimum(realview(Q); kw...)
    MPI.Allreduce(cpuify(locreduce), min, Q.mpicomm)
end
function Base.mapreduce(
    ::typeof(identity),
    ::typeof(max),
    Q::MPIStateArray;
    kw...,
)
    locreduce = maximum(realview(Q); kw...)
    MPI.Allreduce(cpuify(locreduce), max, Q.mpicomm)
end

# helpers: `array_device` and `realview`
array_device(::Union{Array, SArray, MArray}) = CPU()
array_device(::CuArray) = CUDADevice()
array_device(s::SubArray) = array_device(parent(s))
array_device(Q::MPIStateArray) = array_device(Q.data)
array_device(ra::Base.ReshapedArray{T, N, A}) where {T, N, A <: MPIStateArray} =
    array_device(parent(ra))

realview(Q::Union{Array, SArray, MArray}) = Q
realview(Q::MPIStateArray) = Q.realdata
realview(Q::CuArray) = Q

# transform all arguments of `bc` from MPIStateArrays to CuArrays
# and replace CPU function with GPU variants
function transform_broadcasted(bc::Broadcasted, ::CuArray)
    transform_cuarray(bc)
end
function transform_cuarray(bc::Broadcasted)
    Broadcasted(CUDA.cufunc(bc.f), transform_cuarray.(bc.args), bc.axes)
end
transform_cuarray(mpisa::MPIStateArray) = mpisa.realdata
transform_cuarray(x) = x

# @init tictoc()

using KernelAbstractions.Extras: @unroll

@kernel function kernel_fillsendbuf!(
    ::Val{Np},
    ::Val{nvar},
    sendbuf,
    buf,
    vmapsend,
    nvmapsend,
) where {Np, nvar}

    i = @index(Global, Linear)
    @inbounds begin
        e, n = fldmod1(vmapsend[i], Np)
        @unroll for s in 1:nvar
            sendbuf[s, i] = buf[n, s, e]
        end
    end
end

@kernel function kernel_transferrecvbuf!(
    ::Val{Np},
    ::Val{nvar},
    buf,
    recvbuf,
    vmaprecv,
    nvmaprecv,
) where {Np, nvar}

    i = @index(Global, Linear)
    @inbounds begin
        e, n = fldmod1(vmaprecv[i], Np)
        @unroll for s in 1:nvar
            buf[n, s, e] = recvbuf[s, i]
        end
    end
end

bracket_index(i::Int) = "[$i]"
bracket_index(i) = string(i)
"""
    show_not_finite_fields(Q::MPIStateArray)

Prints a warning of which fields are not finite.

!!! warn
    This is an expensive method, which calls `mapreduce`
    on `Q`.
"""
function show_not_finite_fields(Q::MPIStateArray)
    vs = vars(Q)
    not_finite_fields = []
    not_finite =
        Array(reshape(mapreduce(x -> !isfinite(x), |, Q; dims = (1, 3)), :))
    for ftc in flattened_tup_chain(vs)
        i_vars = varsindex(vs, wrap_val.(ftc)...)
        if not_finite[i_vars[1]]
            s = join(bracket_index.(ftc), ".")
            s = replace(s, ".[" => "[")
            push!(not_finite_fields, s)
        end
    end
    if !isempty(not_finite_fields)
        @warn "Field(s) ($(join(not_finite_fields, ", ", ", and "))) are not finite (has NaNs or Inf)"
        flush(stdout)
    end
end

"""
    checked_wait(device, event, progress = nothing, check = false)

If `check` is `false`, simply perform a `wait(device, event, progress)`,
otherwise, check for exceptions and synchronize with all other ranks, so
that all throw an exception.
"""
function checked_wait(device::CPU, event, progress = nothing, check = false)
    err = false
    try
        wait(device, event, progress)
        if check
            err = MPI.Allreduce(false, MPI.MAX, MPI.COMM_WORLD)
        end
    catch exc
        if check
            err = MPI.Allreduce(true, MPI.MAX, MPI.COMM_WORLD)
        end
        rethrow()
    finally
        if check && err
            throw(ErrorOnRemoteNode())
        end
    end
end
function checked_wait(
    device::CUDADevice,
    event,
    progress = nothing,
    check = false,
)
    wait(device, event, progress)
end

end
