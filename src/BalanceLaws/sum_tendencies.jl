##### Sum wrapper

"""
    TENDENCIES_NANS_TRAP()

Set `TENDENCIES_NANS_TRAP() = true`
to throw errors if any tendencies are
not finite.
"""
TENDENCIES_NANS_TRAP() = false

export Σfluxes, Σsources

"""
    flux

An individual flux.
See [`BalanceLaw`](@ref) for more info.
"""
function flux end

"""
    source

An individual source.
See [`BalanceLaw`](@ref) for more info.
"""
function source end

"""
    ntuple_sum(nt::NTuple{N,T}) where {N, T}

sum of `NTuple`, which requires more strict
type input than `sum`. This is added to better
synchronize the success/failure between CPU/GPU
runs to help improve debugging.
"""
ntuple_sum(nt::NTuple{N, T}) where {N, T} = sum(nt)

"""
    Σfluxes(fluxes::NTuple, bl, args)

Sum of the fluxes where
 - `fluxes` is an `NTuple{N, TendencyDef{Flux{O}}} where {N, O}`
 - `bl` is the balance law
 - `args` are the arguments passed to the individual `flux` functions
"""
function Σfluxes(
    pv::PV,
    fluxes::NTuple{N, TendencyDef{Flux{O}}},
    bl,
    args,
) where {N, O, PV}
    vals = ntuple(Val(N)) do i
        projection(pv, bl, fluxes[i], args, flux(pv, fluxes[i], bl, args))
    end
    if TENDENCIES_NANS_TRAP()
        if !all(isfinite.(vals))
            nt = (; zip(fluxes, vals)...)
            println("--------- Non-finite flux found!")
            for (k, v) in nt
                println("Flux: $(string(k)), value = $v")
            end
            error("Non-finite flux found, sorry :(")
        end
    end
    return ntuple_sum(vals)
end

# Emptry scalar case:
function Σfluxes(
    pv::PV,
    fluxes::NTuple{0, TendencyDef{Flux{O}}},
    args...,
) where {O, PV}
    return SVector(0, 0, 0)
end

# Emptry vector case:
function Σfluxes(
    pv::PV,
    fluxes::NTuple{0, TendencyDef{Flux{O}}},
    args...,
) where {O, PV <: AbstractMomentumVariable}
    return SArray{Tuple{3, 3}}(ntuple(i -> 0, 9))
end

# Emptry tracer case:
function Σfluxes(
    pv::PV,
    fluxes::NTuple{0, TendencyDef{Flux{O}}},
    args...,
) where {O, N, PV <: AbstractTracersVariable{N}}
    return SArray{Tuple{3, N}}(ntuple(i -> 0, 3 * N))
end

"""
    Σsources(sources::NTuple, bl, args)

Sum of the sources where
 - `sources` is an `NTuple{N, TendencyDef{Source}} where {N}`
 - `bl` is the balance law
 - `args` are the arguments passed to the individual `source` functions
"""
function Σsources(
    pv::PV,
    sources::NTuple{N, TendencyDef{Source}},
    bl,
    args,
) where {N, PV}
    vals = ntuple(Val(N)) do i
        projection(pv, bl, sources[i], args, source(pv, sources[i], bl, args))
    end
    if TENDENCIES_NANS_TRAP()
        if !all(isfinite.(vals))
            nt = (; zip(sources, vals)...)
            println("--------- Non-finite source found!")
            for (k, v) in nt
                println("Source: $(string(k)), value = $v")
            end
            error("Non-finite source found, sorry :(")
        end
    end
    return ntuple_sum(vals)
end

# Emptry scalar case:
function Σsources(
    pv::PV,
    sources::NTuple{0, TendencyDef{Source}},
    args...,
) where {PV}
    return 0
end

# Emptry vector case:
function Σsources(
    pv::AbstractMomentumVariable,
    sources::NTuple{0, TendencyDef{Source}},
    args...,
)
    return SVector(0, 0, 0)
end

# Emptry tracer case:
function Σsources(
    pv::AbstractTracersVariable{N},
    sources::NTuple{0, TendencyDef{Source}},
    args...,
) where {N}
    return SArray{Tuple{N}}(ntuple(i -> 0, N))
end
