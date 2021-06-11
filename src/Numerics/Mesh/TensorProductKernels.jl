using KernelAbstractions.Extras: @unroll
# This wrapper for CPU and GPU kernels computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
function tpxv!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    ::Val{dims},
    max_threads::Int,
) where {FT <: AbstractFloat, dims}

    args = (vin, vin_den, vout, phir, phis, phit, Val(dims))
    # Launching computational kernel
    if typeof(parent(vin)) <: Array # launch CPU version
        tpxv_hex_CPU!(args...)
    else # launch CUDA version
        Nel = size(vin, 2)
        nthr = min(dims[4] * dims[5] * dims[6], max_threads)
        nblk = cld(dims[4] * dims[5] * dims[6], nthr)
        @cuda threads = (nthr,) blocks = (nblk, Nel) tpxv_hex_CUDA!(args...)
    end

    return nothing
end
# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
# using Julia native multi-threading

function tpxv_hex_CPU!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    ::Val{dims},
) where {FT <: AbstractFloat, dims}
    Nel = size(vin, 2)
    phir_flg = phir isa AbstractArray
    phis_flg = phis isa AbstractArray
    phit_flg = phit isa AbstractArray
    vin_den_flg = vin_den isa AbstractArray

    si, sj, sk = dims[1], dims[2], dims[3] # reading dimensions
    sr, ss, st = dims[4], dims[5], dims[6]

    Threads.@threads for e in 1:Nel
        @inbounds for t in 1:st
            if phit_flg
                k_st, k_en = 1, sk
            else
                k_st, k_en = t, t
            end

            @inbounds for s in 1:ss
                if phis_flg
                    j_st, j_en = 1, sj
                else
                    j_st, j_en = s, s
                end
                @inbounds for r in 1:sr
                    tot = -FT(0)

                    @inbounds for k in k_st:k_en
                        t_rsk = -FT(0)
                        @inbounds for j in j_st:j_en
                            t_rjk = -FT(0)
                            rjk = r + ((j - 1) + (k - 1) * sk) * sr
                            #-----Apply phir-------------
                            if phir_flg
                                if vin_den_flg
                                    @inbounds for i in 1:si
                                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                                        t_rjk +=
                                            phir[r, i] * vin[ijk, e] /
                                            vin_den[ijk, e]
                                    end
                                else
                                    @inbounds for i in 1:si
                                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                                        t_rjk += phir[r, i] * vin[ijk, e]
                                    end
                                end
                            else
                                ijk = r + ((j - 1) + (k - 1) * sj) * si
                                if vin_den_flg
                                    t_rjk = vin[ijk, e] / vin_den[ijk, e]
                                else
                                    t_rjk = vin[ijk, e]
                                end
                            end
                            #-------Apply phis-----------
                            if phis_flg
                                t_rsk += phis[s, j] * t_rjk
                            else
                                t_rsk = t_rjk
                            end
                        end
                        #---Apply phit
                        if phit_flg
                            tot += phit[t, k] * t_rsk
                        else
                            tot = t_rsk
                        end
                    end
                    rst = r + ((s - 1) + (t - 1) * ss) * sr
                    vout[rst, e] = tot
                end
            end
        end
    end
    return nothing
end

# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
# voutᵣₛₜ = phirᵣᵢ phisₛⱼ phitₜₖ vinᵢⱼₖ
# using CUDA kernel
# CUDA kernels are faster for light-weight GPU kernels

function tpxv_hex_CUDA!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    ::Val{dims},
) where {FT <: AbstractFloat, dims}

    tid = threadIdx().x     # local thread id
    bid = blockIdx().x      # block id
    bld = blockDim().x      # block dimension
    e = blockIdx().y      # one element per block
    gid = (bid - 1) * bld + tid # global thread id

    si, sj, sk = dims[1], dims[2], dims[3] # reading dimensions
    sr, ss, st = dims[4], dims[5], dims[6]

    r = gid % sr == 0 ? sr : gid % sr                   # r for each thread
    s = cld(gid, sr) % ss == 0 ? ss : cld(gid, sr) % ss # s for each thread
    t = cld(gid, sr * ss)                                 # t for each thread

    if phir isa AbstractArray
        lphir = MVector{si, FT}(undef)
        @inbounds @unroll for i in 1:si
            lphir[i] = phir[r, i]
        end
    end

    if phis isa AbstractArray
        j_st, j_en = 1, sj
    else
        j_st, j_en = s, s
    end

    if phit isa AbstractArray
        k_st, k_en = 1, sk
    else
        k_st, k_en = t, t
    end

    tot = -FT(0)

    @inbounds for k in k_st:k_en
        t_rsk = -FT(0)
        @inbounds for j in j_st:j_en
            t_rjk = -FT(0)
            rjk = r + ((j - 1) + (k - 1) * sk) * sr
            #-----Apply phir-------------
            if phir isa AbstractArray
                if vin_den isa AbstractArray
                    @inbounds @unroll for i in 1:si
                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                        t_rjk += lphir[i] * vin[ijk, e] / vin_den[ijk, e]
                    end
                else
                    @inbounds @unroll for i in 1:si
                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                        t_rjk += lphir[i] * vin[ijk, e]
                    end
                end
            else
                ijk = r + ((j - 1) + (k - 1) * sj) * si
                if vin_den isa AbstractArray
                    @inbounds t_rjk = vin[ijk, e] / vin_den[ijk, e]
                else
                    @inbounds t_rjk = vin[ijk, e]
                end
            end
            #-------Apply phis-----------
            if phis isa AbstractArray
                @inbounds t_rsk += phis[s, j] * t_rjk
            else
                t_rsk = t_rjk
            end
        end
        #---Apply phit
        if phit isa AbstractArray
            @inbounds tot += phit[t, k] * t_rsk
        else
            tot = t_rsk
        end
    end
    rst = r + ((s - 1) + (t - 1) * ss) * sr
    @inbounds vout[rst, e] = tot

    return nothing
end
