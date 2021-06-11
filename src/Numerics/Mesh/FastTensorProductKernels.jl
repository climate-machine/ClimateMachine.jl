# This wrapper for CPU and GPU kernels computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
# voutᵣₛₜ = phirᵣᵢ  phisₛⱼ phitₜₖ vinᵢⱼₖ

function ftpxv!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    ::Val{dims},
    temp::AbstractArray{FT, 3},
    max_threads::Int,
) where {FT <: AbstractFloat, dims}

    args = (vin, vin_den, vout, phir, phis, phit, Val(dims), temp)
    # Launching computational kernel
    if typeof(parent(vin)) <: Array
        ftpxv_hex_CPU!(args...)
    else # CUDA version
        Nel = size(vin, 2)
        @cuda threads = (max_threads,) blocks = (Nel,) ftpxv_hex_CUDA!(args...)
    end
    return nothing
end

# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
# using Julia native multi-threading

function ftpxv_hex_CPU!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    ::Val{dims},
    temp::AbstractArray{FT, 3},
) where {FT <: AbstractFloat, dims}
    Nel = size(vin, 2)
    si, sj, sk, sr, ss, st = dims # reading dimensions

    dflg = vin_den isa AbstractArray
    rflg = phir isa AbstractArray
    sflg = phis isa AbstractArray
    tflg = phit isa AbstractArray

    Threads.@threads for e in 1:Nel
        #--------------------------------------------------------------------------------
        if rflg && !sflg && !tflg
            #----Apply phir
            for k in 1:sk, j in 1:sj, r in 1:sr
                tot = -FT(0)
                rst = r + ((j - 1) + (k - 1) * ss) * sr
                if dflg
                    for i in 1:si
                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                        tot += phir[r, i] * vin[ijk, e] / vin_den[ijk, e]
                    end
                    vout[rst, e] = tot
                else
                    for i in 1:si
                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                        tot += phir[r, i] * vin[ijk, e]
                    end
                    vout[rst, e] = tot
                end
            end
            #-------------------------------------------------------------
        elseif !rflg && sflg && !tflg
            #----Apply phis
            for k in 1:sk, s in 1:ss, r in 1:sr
                tot = -FT(0)
                rst = r + ((s - 1) + (k - 1) * ss) * sr
                if dflg
                    for j in 1:sj
                        rjk = r + ((j - 1) + (k - 1) * sj) * sr
                        tot += phis[s, j] * vin[rjk, e] / vin_den[rjk, e]
                    end
                    vout[rst, e] = tot
                else
                    for j in 1:sj
                        rjk = r + ((j - 1) + (k - 1) * sj) * sr
                        tot += phis[s, j] * vin[rjk, e]
                    end
                    vout[rst, e] = tot
                end
            end
            #-------------------------------------------------------------
        elseif !rflg && !sflg && tflg
            #----Apply phit
            for t in 1:st, s in 1:ss, r in 1:sr
                tot = -FT(0)
                rst = r + ((s - 1) + (t - 1) * ss) * sr
                if dflg
                    for k in 1:sk
                        rsk = r + ((s - 1) + (k - 1) * ss) * sr
                        tot += phit[t, k] * vin[rsk, e] / vin_den[rsk, e]
                    end
                    vout[rst, e] = tot
                else
                    for k in 1:sk
                        rsk = r + ((s - 1) + (k - 1) * ss) * sr
                        tot += phit[t, k] * vin[rsk, e]
                    end
                    vout[rst, e] = tot
                end
            end
            #-------------------------------------------------------------
        elseif rflg && sflg && tflg
            # apply phir
            for k in 1:sk, j in 1:sj, r in 1:sr
                tot = -FT(0)
                if dflg
                    for i in 1:si
                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                        tot += phir[r, i] * vin[ijk, e] / vin_den[ijk, e]
                    end
                    rjk = r + ((j - 1) + (k - 1) * sj) * sr
                    temp[rjk, 1, e] = tot
                else
                    for i in 1:si
                        ijk = i + ((j - 1) + (k - 1) * sj) * si
                        tot += phir[r, i] * vin[ijk, e]
                    end
                    rjk = r + ((j - 1) + (k - 1) * sj) * sr
                    temp[rjk, 1, e] = tot
                end
            end
            # apply phis
            for k in 1:sk, s in 1:ss, r in 1:sr
                tot = -FT(0)
                for j in 1:sj
                    rjk = r + ((j - 1) + (k - 1) * sj) * sr
                    tot += phis[s, j] * temp[rjk, 1, e]
                end
                rsk = r + ((s - 1) + (k - 1) * ss) * sr
                temp[rsk, 2, e] = tot
            end
            # apply phit
            for t in 1:st, s in 1:ss, r in 1:sr
                tot = -FT(0)
                for k in 1:sk
                    rsk = r + ((s - 1) + (k - 1) * ss) * sr
                    tot += phit[t, k] * temp[rsk, 2, e]
                end
                rst = r + ((s - 1) + (t - 1) * ss) * sr
                vout[rst, e] = tot
            end
            #-------------------------------------------------------------
        end
    end
    return nothing
end

# This kernel computes the fast tensor-product x vector
# vout = (phit ⊗ phis ⊗ phir) * vin
# using CUDA kernel
# CUDA kernels are faster for light-weight GPU kernels

function ftpxv_hex_CUDA!(
    vin::AbstractArray{FT, 2},
    vin_den::Union{AbstractArray{FT, 2}, Nothing},
    vout::AbstractArray{FT, 2},
    phir::Union{AbstractArray{FT, 2}, Nothing},
    phis::Union{AbstractArray{FT, 2}, Nothing},
    phit::Union{AbstractArray{FT, 2}, Nothing},
    ::Val{dims},
    temp::AbstractArray{FT, 3},
) where {FT <: AbstractFloat, dims}

    tid = threadIdx().x # thread ids
    e = blockIdx().x    # block id / elem #
    bld = blockDim().x  # block size

    si, sj, sk, sr, ss, st = dims # reading dimensions

    dflg = vin_den isa AbstractArray
    rflg = phir isa AbstractArray
    sflg = phis isa AbstractArray
    tflg = phit isa AbstractArray

    if rflg && !sflg && !tflg
        #----Apply phir
        nloops = cld(sr * sj * sk, bld)
        @inbounds for lp in 1:nloops
            gid = tid + (lp - 1) * bld
            i_r = gid % sr == 0 ? sr : gid % sr
            i_j = cld(gid, sr) % sj == 0 ? sj : cld(gid, sr) % sj
            i_k = cld(gid, sr * sj)
            if i_r ≤ sr && i_j ≤ sj && i_k ≤ sk
                tot = -FT(0)
                @inbounds for i in 1:si
                    ijk = i + ((i_j - 1) + (i_k - 1) * sj) * si
                    if dflg
                        @inbounds tot +=
                            phir[i_r, i] * vin[ijk, e] / vin_den[ijk, e]
                    else
                        @inbounds tot += phir[i_r, i] * vin[ijk, e]
                    end
                end
                rst = i_r + ((i_j - 1) + (i_k - 1) * ss) * sr
                @inbounds vout[rst, e] = tot
            end
        end
    elseif !rflg && sflg && !tflg
        #-----Apply phis
        nloops = cld(sr * ss * sk, bld)
        @inbounds for lp in 1:nloops
            gid = tid + (lp - 1) * bld
            i_r = gid % sr == 0 ? sr : gid % sr
            i_s = cld(gid, sr) % ss == 0 ? ss : cld(gid, sr) % ss
            i_k = cld(gid, sr * ss)
            if i_r ≤ sr && i_s ≤ ss && i_k ≤ sk
                tot = -FT(0)
                @inbounds for j in 1:sj
                    rjk = i_r + ((j - 1) + (i_k - 1) * sj) * sr
                    if dflg
                        @inbounds tot +=
                            phis[i_s, j] * vin[rjk, e] / vin_den[rjk, e]
                    else
                        @inbounds tot += phis[i_s, j] * vin[rjk, e]
                    end
                end
                rst = i_r + ((i_s - 1) + (i_k - 1) * ss) * sr
                @inbounds vout[rst, e] = tot
            end
        end
    elseif !rflg && !sflg && tflg
        #-----Apply phit
        nloops = cld(sr * ss * st, bld)
        @inbounds for lp in 1:nloops
            gid = tid + (lp - 1) * bld
            i_r = gid % sr == 0 ? sr : gid % sr
            i_s = cld(gid, sr) % ss == 0 ? ss : cld(gid, sr) % ss
            i_t = cld(gid, sr * ss)
            if i_r ≤ sr && i_s ≤ ss && i_t ≤ st
                tot = -FT(0)
                @inbounds for k in 1:sk
                    rsk = i_r + ((i_s - 1) + (k - 1) * ss) * sr
                    if dflg
                        @inbounds tot +=
                            phit[i_t, k] * vin[rsk, e] / vin_den[rsk, e]
                    else
                        @inbounds tot += phit[i_t, k] * vin[rsk, e]
                    end
                end
                rst = i_r + ((i_s - 1) + (i_t - 1) * ss) * sr
                @inbounds vout[rst, e] = tot
            end
        end
    elseif rflg && sflg && tflg
        #----Apply phir
        nloops = cld(sr * sj * sk, bld)
        @inbounds for lp in 1:nloops
            gid = tid + (lp - 1) * bld
            i_r = gid % sr == 0 ? sr : gid % sr
            i_j = cld(gid, sr) % sj == 0 ? sj : cld(gid, sr) % sj
            i_k = cld(gid, sr * sj)
            if i_r ≤ sr && i_j ≤ sj && i_k ≤ sk
                tot = -FT(0)
                if dflg
                    @inbounds for i in 1:si
                        ijk = i + ((i_j - 1) + (i_k - 1) * sj) * si
                        @inbounds tot +=
                            phir[i_r, i] * vin[ijk, e] / vin_den[ijk, e]
                    end
                else
                    @inbounds for i in 1:si
                        ijk = i + ((i_j - 1) + (i_k - 1) * sj) * si
                        @inbounds tot += phir[i_r, i] * vin[ijk, e]
                    end
                end
                rjk = i_r + ((i_j - 1) + (i_k - 1) * sj) * sr
                @inbounds temp[rjk, 1, e] = tot
            end
        end
        sync_threads()
        #-----Apply phis
        nloops = cld(sr * ss * sk, bld)
        @inbounds for lp in 1:nloops
            gid = tid + (lp - 1) * bld
            i_r = gid % sr == 0 ? sr : gid % sr
            i_s = cld(gid, sr) % ss == 0 ? ss : cld(gid, sr) % ss
            i_k = cld(gid, sr * ss)
            if i_r ≤ sr && i_s ≤ ss && i_k ≤ sk
                tot = -FT(0)
                @inbounds for j in 1:sj
                    rjk = i_r + ((j - 1) + (i_k - 1) * sj) * sr
                    @inbounds tot += phis[i_s, j] * temp[rjk, 1, e]
                end
                rsk = i_r + ((i_s - 1) + (i_k - 1) * ss) * sr
                @inbounds temp[rsk, 2, e] = tot
            end
        end
        sync_threads()
        #-----Apply phit
        nloops = cld(sr * ss * st, bld)
        @inbounds for lp in 1:nloops
            gid = tid + (lp - 1) * bld
            i_r = gid % sr == 0 ? sr : gid % sr
            i_s = cld(gid, sr) % ss == 0 ? ss : cld(gid, sr) % ss
            i_t = cld(gid, sr * ss)
            if i_r ≤ sr && i_s ≤ ss && i_t ≤ st
                tot = -FT(0)
                @inbounds for k in 1:sk
                    rsk = i_r + ((i_s - 1) + (k - 1) * ss) * sr
                    @inbounds tot += phit[i_t, k] * temp[rsk, 2, e]
                end
                rst = i_r + ((i_s - 1) + (i_t - 1) * ss) * sr
                @inbounds vout[rst, e] = tot
            end
        end
    end

    return nothing
end
