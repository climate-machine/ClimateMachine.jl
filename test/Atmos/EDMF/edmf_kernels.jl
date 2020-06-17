#### EDMF model kernels

include(joinpath("helper_funcs", "nondimensional_exchange_functions.jl"))
include(joinpath("helper_funcs", "lamb_smooth_minimum.jl"))
include(joinpath("helper_funcs", "subdomain_statistics.jl"))
include(joinpath("closures", "entr_detr.jl"))
include(joinpath("closures", "pressure.jl"))
include(joinpath("closures", "mixing_length.jl"))
include(joinpath("closures", "turbulence_functions.jl"))
include(joinpath("closures", "surface_functions.jl"))
# include(joinpath("closures", "micro_phys.jl"))

function vars_state_auxiliary(m::NTuple{N, Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_auxiliary(m[i], FT), N)...}
end

function vars_state_auxiliary(::Updraft, FT)
    @vars(buoyancy::FT,
          updraft_top::FT,# YAIR - DO I NEED THIS 
          )
end

function vars_state_auxiliary(::Environment, FT)
    @vars(cld_frac::FT,# YAIR - DO I NEED THIS 
          )
end

function vars_state_auxiliary(m::EDMF, FT)
    @vars(environment::vars_state_auxiliary(m.environment, FT),
          updraft::vars_state_auxiliary(m.updraft, FT)
          );
end

function vars_state_conservative(::Updraft, FT)
    @vars(ρa::FT,
          ρau::SVector{3, FT},
          ρae::FT,
          ρaq_tot::FT,
          )
end

function vars_state_conservative(::Environment, FT)
    @vars(ρatke::FT,
          ρae_cv::FT,
          ρaq_tot_cv::FT,
          ρae_q_tot_cv::FT,
          )
end

function vars_state_conservative(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_conservative(m[i], FT), N)...}
end


function vars_state_conservative(m::EDMF, FT)
    @vars(environment::vars_state_conservative(m.environment, FT),
          updraft::vars_state_conservative(m.updraft, FT)
          );
end

function vars_state_gradient(::Updraft, FT)
    @vars(u::SVector{3, FT},
         );
end

function vars_state_gradient(::Environment, FT)
    @vars(e::FT,
          e_int::FT,
          q_tot::FT,
          u::SVector{3, FT},
          tke::FT,
          e_cv::FT,
          q_tot_cv::FT,
          e_q_tot_cv::FT,
          θ_ρ::FT,
          );
end


function vars_state_gradient(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_gradient(m[i], FT), N)...}
end


function vars_state_gradient(m::EDMF, FT)
    @vars(environment::vars_state_gradient(m.environment, FT),
          updraft::vars_state_gradient(m.updraft, FT)
          );
end


function vars_state_gradient_flux(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_gradient_flux(m[i], FT), N)...}
end

function vars_state_gradient_flux(::Updraft, FT)
    @vars(∇u::SMatrix{3, 3, FT, 9},
          );
end

function vars_state_gradient_flux(::Environment, FT)
    @vars(∇e::SVector{3, FT},
          ∇e_int::SVector{3, FT},
          ∇q_tot::SVector{3, FT},
          ∇u::SMatrix{3, 3, FT, 9},
          ∇tke::SVector{3, FT},
          ∇e_cv::SVector{3, FT},
          ∇q_tot_cv::SVector{3, FT},
          ∇e_q_tot_cv::SVector{3, FT},
          ∇θ_ρ::SVector{3, FT},# used in a diagnostic equation for the mixing length
          );

end

function vars_state_gradient_flux(m::EDMF, FT)
    @vars(environment::vars_state_gradient_flux(m.environment, FT),
          updraft::vars_state_gradient_flux(m.updraft, FT)
          );
end

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state_auxiliary`
function init_state_auxiliary!(
        m::SingleStack{FT,N},
        edmf::EDMF{FT,N},
        aux::Vars,
        geom::LocalGeometry
    ) where {FT,N}
    # Aliases:
    en_a = aux.edmf.environment
    up_a = aux.edmf.updraft

    # en_a.buoyancy = eps(FT)
    en_a.cld_frac = eps(FT)

    for i in 1:N
        up_a[i].buoyancy = eps(FT)
        up_a[i].updraft_top  = eps(FT)
    end
    en_a.cld_frac = eps(FT)

end;

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
function init_state_conservative!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
) where {FT,N}

    # Aliases:
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft

    # GCM setting - Initialize the grid mean profiles of prognostic variables (ρ,e,q_tot,u,v,w)
    # a_up = m.a_updraft_initial/FT(N)
    a_up = 0.1/FT(N)
    for i in 1:N
        up[i].ρa = gm.ρ * a_up
        up[i].ρau = gm.ρu * a_up
        up[i].ρae = gm.ρe * a_up
        up[i].ρaq_tot = gm.ρq_tot * a_up
    end

    # initialize environment covariance
    en.ρae_cv       = eps(FT)
    en.ρaq_tot_cv       = eps(FT)
    en.ρae_q_tot_cv = eps(FT)

end;

# The remaining methods, defined in this section, are called at every
# time-step in the solver by the [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `single_stack_nodal_update_aux!`, or
# any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::SingleStack{FT, N},
    edmf::EDMF,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
) where {FT, N}

    # Move to backend (requires `along_stack(max, var_name::Symbol)`):
    state_vars = SingleStackUtils.get_vars_from_nodal_stack(
        dg.grid,
        Q,
        vars_state_conservative(m, FT),
    )
    aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
        dg.grid,
        dg.state_auxiliary,
        vars_state_auxiliary(m, FT),
    )
    z = aux_vars["z"]

    # Ideally, it looks something like this:
    # for i in 1:N
    #     edmf.updraft[i].updraft_top, z_loc = along_stack(max(edmf.updraft[i].ρa/state.ρ, 0), Q, elems)
    # end

    for i in 1:N
        for j in 1:length(z)
            up_i_ρa = state_vars["edmf.updraft[$i].ρa"][j]
            ρinv = 1/state_vars["ρ"][j]
            if up_i_ρa*ρinv>0
                aux_vars["edmf.updraft[$i].updraft_top"][j] = z[j]
            end
        end
        # i_updraft_top = varsindex(vars_state_conservative(m, FT), :updraft_top)
        # Q[ijk, i_updraft_top, elems] = aux_vars["edmf.updraft[$i].updraft_top"][j]
    end

    nodal_update_auxiliary_state!(edmf_stack_nodal_update_aux!, dg, m, Q, t, elems)
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in
# `vars_state_auxiliary`
function edmf_stack_nodal_update_aux!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}

    en_a = aux.edmf.environment
    up_a = aux.edmf.updraft
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft

    #   -------------  Compute buoyancies of subdomains
    ρinv = 1/gm.ρ
    _grav::FT = grav(m.param_set)
    # b_upds = 0
    # a_upds = 0
    for i in 1:N
        # override ----------------
        # e_int_up = internal_energy(gm.ρ, up[i].ρae/up[i].ρa, up[i].ρau/up[i].ρa, _grav * gm_a.z)
        # ts = PhaseEquil(m.param_set ,e_int_up, gm.ρ, up[i].ρaq_tot/up[i].ρa)
        # ρ_i = #air_density(ts)
        # override ----------------
        ρ_i = aux.ρ0
        up_a[i].buoyancy = -_grav*(ρ_i-aux.ρ0)*ρinv
    end
    # compute the buoyancy of the environment
    en_area   = 1 - sum([up[i].ρa for i in 1:N])*ρinv
    env_e     = (gm.ρe     - sum( [up[i].ρae*ρinv for i in 1:N] ))/en_area
    env_q_tot = (gm.ρq_tot - sum( [up[i].ρaq_tot*ρinv for i in 1:N] ))/en_area 

    # override ----------------
    # ts = PhaseEquil(m.param_set ,env_e, gm.ρ, env_q_tot)
    # env_ρ = air_density(ts)
    # env_q_liq = PhasePartition(ts).liq
    # env_q_ice = PhasePartition(ts).ice

    env_ρ = aux.ρ0
    env_q_liq = FT(0)
    env_q_ice = FT(0)
    # override ----------------

    b_env = -_grav*(env_ρ - aux.ρ0)*ρinv
    b_gm = en_area*b_env + sum([up_a[i].buoyancy for i in 1:N])
    # subtract the grid mean
    for i in 1:N
        up_a[i].buoyancy -= b_gm
    end

    #   -------------  Compute upd_top
    # YAIR - check this with Charlie
    for i in 1:N
        for j in length(up[i].ρa)
            if up[i].ρa*ρinv>eps(FT)
                up_a[i].updraft_top = aux.z
            end
        end
    end
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
# the gradient of `ρcT`. Here, we specify how `ρcT` is computed. Note that
#  - `transform.ρcT` is available here because we've specified `ρcT` in
#  `vars_state_gradient`
function compute_gradient_argument!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    # Aliases:
    up_t = transform.edmf.updraft
    en_t = transform.edmf.environment
    gm = state
    up = state.edmf.updraft
    en = state.edmf.environment

    up_t.u = up.ρau/up.ρa
    _grav = grav(m.param_set)
    ts = thermo_state(SingleStack, state, aux)

    ρinv       = 1/gm.ρ
    en_area    = 1 - sum([up[i].ρa for i in 1:N])*ρinv

    en_ρe = (gm.ρe-sum([up[j].ρae for j in 1:N]))/a_en
    en_ρu = (gm.ρu-sum([up[j].ρae for j in 1:N]))/a_en
    en_e_int = internal_energy(gm.ρ, en_ρe, en_ρu, _grav * gm_a.z)
    
    # populate gradient arguments
    en_t.e     = en_ρe*ρinv
    en_t.e_int = en_e_int
    en_t.q_tot = (gm.ρq_tot - sum([up[i].ρaq_tot for i in 1:N]))/(en_area*gm.ρ)
    en_t.u     = (gm.ρu - sum([up[i].ρau for i in 1:N]))/(en_area*gm.ρ)

    en_t.tke        = en.ρatke/(en_area*gm.ρ)
    en_t.e_cv       = en.ρae_cv/(en_area*gm.ρ)
    en_t.q_tot_cv   = en.ρaq_tot_cv/(en_area*gm.ρ)
    en_t.e_q_tot_cv = en.ρae_q_tot_cv/(en_area*gm.ρ)

    en_t.θ_ρ = virtual_pottemp(ts)
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
function compute_gradient_flux!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    # Aliases:
    gm = state
    gm_d = diffusive
    up_d = diffusive.edmf.updraft
    up_∇t = ∇transform.edmf.updraft
    en_d = diffusive.edmf.environment
    en_∇t = ∇transform.edmf.environment

    for i in 1:N
        up_d[i].∇u = up_∇t[i].u
    end

    ρinv      = 1/gm.ρ
    # negative signs here as we have a '-' sign in BL form leading to + K∂ϕ/∂z on the RHS
    # first moment grid mean comming from enviroment gradients only
    en_d.∇e     = en_∇t.e
    en_d.∇e_int = en_∇t.e_int
    en_d.∇q_tot = en_∇t.q_tot
    en_d.∇u     = en_∇t.u
    # second moment env cov
    en_d.∇tke        = en_∇t.tke
    en_d.∇e_cv       = en_∇t.e_cv
    en_d.∇q_tot_cv   = en_∇t.q_tot_cv
    en_d.∇e_q_tot_cv = en_∇t.e_q_tot_cv

    en_d.∇θ_ρ = en_∇t.θ_ρ

end;

# We have no sources, nor non-diffusive fluxes.
function source!(
    m::SingleStack{FT, N},
    edmf::EDMF{FT, N},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) where {FT, N}

    # Aliases:
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft
    gm_s = state
    en_s = state.edmf.environment
    up_s = state.edmf.updraft
    en_d = diffusive.edmf.environment


    # grid mean sources - I think that large scale subsidence in
    #            doubly periodic domains should be applied here
    # updraft sources

    # YAIR  - these need to be defined as vectors length N - check with Charlie
    εt = MArray{Tuple{N}, FT}(zeros(FT, N))
    ε = MArray{Tuple{N}, FT}(zeros(FT, N))
    δ = MArray{Tuple{N}, FT}(zeros(FT, N))
    # should be conditioned on updraft_area > minval
    
    # get environment values for e, q_tot , u[3]
    ρinv = 1/gm.ρ
    a_env = 1 - sum([up[i].ρa for i in 1:N])*ρinv
    w_env = (gm.ρu[3] - sum([up[i].ρau[3] for i in 1:N]))*ρinv
    e_env = (gm.ρe - sum([up[i].ρae for i in 1:N]))*ρinv
    q_tot_env = (gm.ρq_tot - sum([up[i].ρaq_tot for i in 1:N]))*ρinv
    
    ρinv  = 1/gm.ρ
    for i in 1:N
        # upd vars
        u_env = (gm.ρu - up[i].ρau)/(gm.ρ*a_env)
        w_i               = up[i].ρau[3]/up[i].ρa
        
        # first moment sources
        # ε[i], δ[i], εt[i] = entr_detr(m, edmf.entr_detr, state, aux, t, i)
        # dpdz, dpdz_tke_i  = perturbation_pressure(m, edmf.pressure, state, diffusive, aux, t, direction, i)
        dpdz = FT(0)
        dpdz_tke_i = FT(0)

        # entrainment and detrainment
        up_s[i].ρa      += up[i].ρa * w_i * ( ε[i]                        
                        -  δ[i])
        up_s[i].ρau     += up[i].ρa * w_i * ((ε[i]+εt[i])*up_s[i].ρau     
                        - (δ[i]+εt[i])*u_env)
        up_s[i].ρae     += up[i].ρa * w_i * ((ε[i]+εt[i])*up_s[i].ρae     
                        - (δ[i]+εt[i])*e_env)
        up_s[i].ρaq_tot += up[i].ρa * w_i * ((ε[i]+εt[i])*up_s[i].ρaq_tot 
                        - (δ[i]+εt[i])*q_tot_env)

        # perturbation pressure in w equation
        up_s[i].ρau += [0,0,up[i].ρa * dpdz] # CHARLIE check me on this one

        # microphysics sources should be applied here

        ## environment second moments:

        # covariances entrinament sources from the i'th updraft
        # need to compute e_int in updraft and gridmean for entrainment 
        # -- if ϕ'ψ' is tke and ϕ,ψ are both w than a factor 0.5 appears in the εt and δ terms
        # Covar_Source      +=  ρaw⋅δ⋅(ϕ_up-ϕ_en) ⋅ (ψ_up-ψ_en) 
        #                     + ρaw⋅εt⋅[(ϕ_up-⟨ϕ⟩)⋅(ψ_up-ψ_en) 
        #                     + (ϕ_up-⟨ϕ⟩)⋅(ψ_up-ψ_en)] - ρaw⋅ε⋅ϕ'ψ'

        en_s.ρatke        += (up[i].ρau[3]*δ[i]
                            *(up[i].ρau[3]/up[i].ρa - w_env)
                            *(up[i].ρau[3]/up[i].ρa - w_env)*0.5
                             +up[i].ρau[3]*εt[i]*w_env
                            *(up[i].ρau[3]/up[i].ρa - gm.ρu[i]*ρinv)
                             -up[i].ρau[3]*ε[i] * en.ρatke)

        en_s.ρae_cv       += (up[i].ρau[3]*δ[i] 
                            *(up[i].ρae/up[i].ρa - e_env)
                            *(up[i].ρae/up[i].ρa - e_env)
                             +up[i].ρau[3]*εt[i]*e_env
                             *(up[i].ρae/up[i].ρa - gm.ρe*ρinv)*2
                             -up[i].ρau[3]*ε[i]*en.ρae_cv)

        en_s.ρaq_tot_cv   += (up[i].ρau[3]*δ[i]
                            *(up[i].ρaq_tot/up[i].ρa - q_tot_env)
                            *(up[i].ρaq_tot/up[i].ρa - q_tot_env)
                            + up[i].ρau[3]*εt[i]*q_tot_env
                            *(up[i].ρaq_tot/up[i].ρa - gm.ρq_tot*ρinv)*2
                            - up[i].ρau[3]*ε[i] *en.ρaq_tot_cv)

        en_s.ρae_q_tot_cv += (up[i].ρau[3]*δ[i] 
                            *(up[i].ρae/up[i].ρa - e_env)
                            *(up[i].ρaq_tot/up[i].ρa - q_tot_env)
                            + up[i].ρau[3]*εt[i]*e_env
                            *(up[i].ρaq_tot/up[i].ρa - gm.ρq_tot*ρinv)
                            + up[i].ρau[3]*εt[i]
                            *q_tot_env*(up[i].ρae/up[i].ρa - gm.ρe*ρinv)
                            - up[i].ρau[3]*ε[i] *en.ρae_q_tot_cv)
        
        # pressure tke source from the i'th updraft
        en_s.ρatke += up[i].ρa * dpdz_tke_i
    end
    # mix_len    = mixing_length(m, m.edmf.mix_len, state, diffusive, aux, t, δ, εt)
    mix_len = FT(0)
    en_tke = en.ρatke*ρinv/a_env
    K_eddy = m.edmf.mix_len.c_k*mix_len*sqrt(en_tke)
    Shear  = en_d.∇u[1].^2 + en_d.∇u[2].^2 + en_d.∇u[3].^2 # consider scalar product of two vectors

    # second moment production from mean gradients (+ sign here as we have + S in BL form)
    #                            production from mean gradient           - Dissipation
    en_s.ρatke        += gm.ρ*a_env*K_eddy*Shear                         
                       - edmf.mix_len.c_m*sqrt(en_tke)/mix_len*en.ρatke
    en_s.ρae_cv       += gm.ρ*a_env*K_eddy*en_d.∇e[3]*en_d.∇e[3] 
                       - edmf.mix_len.c_m*sqrt(en_tke)/mix_len*en.ρae_cv
    en_s.ρaq_tot_cv   += gm.ρ*a_env*K_eddy*en_d.∇q_tot[3]*en_d.∇q_tot[3] 
                       - edmf.mix_len.c_m*sqrt(en_tke)/mix_len*en.ρaq_tot_cv
    en_s.ρae_q_tot_cv += gm.ρ*a_env*K_eddy*en_d.∇e[3]*en_d.∇q_tot[3] 
                       - edmf.mix_len.c_m*sqrt(en_tke)/mix_len*en.ρae_q_tot_cv
    # covariance microphysics sources should be applied here
end;

# in the EDMF first order (advective) fluxes exist only in the grid mean (if <w> is nonzero) and the uprdafts
function flux_first_order!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT, N},
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}

    # Aliases:
    gm = state
    up = state.edmf.updraft
    up_f = flux.edmf.updraft

    # positive sign here as we have a '-' sign in BL form leading to - ∂ρwϕ/∂z on the RHS
    # updrafts
    ρinv = 1/gm.ρ
    for i in 1:N
        up_f[i].ρa = up[i].ρau
        u = up[i].ρau / up[i].ρa
        up_f[i].ρau = up[i].ρau * u'
        up_f[i].ρae = u * up[i].ρae
        up_f[i].ρaq_tot = u * up[i].ρaq_tot
    end

end;

# in the EDMF second order (diffusive) fluxes exist only in the grid mean and the enviroment
function flux_second_order!(
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    # Aliases:
    gm   = state
    up   = state.edmf.updraft
    en   = state.edmf.environment
    gm_f = flux
    up_f = flux.edmf.updraft
    en_f = flux.edmf.environment
    en_d = diffusive.edmf.environment

    ρinv = FT(1)/gm.ρ
    en_ρa = gm.ρ-sum([up[i].ρa for i in 1:N])

    εt = MArray{Tuple{N}, FT}(zeros(FT, N))
    ε = MArray{Tuple{N}, FT}(zeros(FT, N))
    δ = MArray{Tuple{N}, FT}(zeros(FT, N))
    for i in 1:N
        # ε[i], δ[i], εt[i] = entr_detr(m, m.edmf.entr_detr, state, aux, t, i)
        ε[i] = FT(0)
        δ[i] = FT(0)
        εt[i] = FT(0)
    end
    # mix_len = mixing_length(m, edmf.mix_len, state, diffusive, aux, t, δ, εt)
    mix_len = FT(0)
    ρa_env = (gm.ρ-sum([up[j].ρa for j in 1:N]))
    K_eddy = m.edmf.mix_len.c_k*mix_len*sqrt(abs(en.ρatke/ρa_env)) # YAIR 

    ## we are adding the massflux term here as it is part of the total flux:
    #total flux(ϕ) =   diffusive_flux(ϕ)  +   massflux(ϕ)
    #   ⟨w ⃰ ϕ ⃰ ⟩   = - a_0 K_eddy⋅∂ϕ/∂z + ∑ a_i(w_i-⟨w⟩)(ϕ_i-⟨ϕ⟩)
    massflux_e     = sum([ up[i].ρa*ρinv*(gm.ρe*ρinv     
        - up[i].ρae/up[i].ρa)*(gm.ρu[3]*ρinv     
        - up[i].ρau[3]/up[i].ρa) for i in 1:N])
    massflux_q_tot = sum([ up[i].ρa*ρinv*(gm.ρq_tot*ρinv 
        - up[i].ρaq_tot/up[i].ρa)*(gm.ρu[3]*ρinv 
        - up[i].ρau[3]/up[i].ρa) for i in 1:N])
    massflux_u     = sum([ up[i].ρa*ρinv*(gm.ρu*ρinv     
        - up[i].ρau/up[i].ρa)*(gm.ρu[3]*ρinv     
        - up[i].ρau[3]/up[i].ρa) for i in 1:N])

    # update grid mean flux_second_order
    gm_f.ρe     += -ρa_env*K_eddy*en_d.∇e[3]     + massflux_e
    gm_f.ρq_tot += -ρa_env*K_eddy*en_d.∇q_tot[3] + massflux_q_tot
    # override ---------------- adding massflux below u breaks
    gm_f.ρu     += -ρa_env.*K_eddy.*en_d.∇u
    # gm_f.ρu     += -ρa_env.*K_eddy.*en_d.∇u      + massflux_u

    # env second momment flux_second_order
    en_f.ρatke        += -ρa_env*K_eddy*en_d.∇tke[3]
    en_f.ρae_cv       += -ρa_env*K_eddy*en_d.∇e_cv[3]
    en_f.ρaq_tot_cv   += -ρa_env*K_eddy*en_d.∇q_tot_cv[3]
    en_f.ρae_q_tot_cv += -ρa_env*K_eddy*en_d.∇e_q_tot_cv[3]
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = α∇ρcT``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρcT` (first order unknown)
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) where {FT,N}
    gm = state⁺
    up = state⁺.edmf.updraft
    if bctype == 1 # bottom
        # YAIR - questions which state should I use here , state⁺ or state⁻  for computation of surface processes
        upd_a_surf, upd_e_surf, upd_q_tot_surf  = compute_updraft_surface_BC(m, m.edmf.surface, edmf, gm)
        upd_a_surf = FT(0)
        upd_e_surf = FT(0)
        upd_q_tot_surf = FT(0)
        for i in 1:N

            up[i].ρau     = SVector(0,0,0)
            up[i].ρa      = FT(0.1)#upd_a_surf[i]
            up[i].ρae     = FT(30000)#upd_e_surf[i]
            up[i].ρaq_tot = eps(FT)#upd_q_tot_surf[i]
        end
        # can call `env_surface_covariances` with surface values

    elseif bctype == 2 # top
        # if yes not BC on upd are needed at the top (currently set to GM)
        # if not many issues might  come up with area fraction at the upd top
        ρinv = 1/gm.ρ
        for i in 1:N
            up[i].ρau = SVector(0,0,0)
            up[i].ρa = FT(0)
            up[i].ρae = FT(0)
            up[i].ρaq_tot = FT(0)
        end

    end
end;

# The boundary conditions for `ρcT` are specified here for second-order
# unknowns
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    edmf::EDMF{FT,N},
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) where {FT,N}
    gm = state⁺
    up = state⁺.edmf.updraft
    en = state⁺.edmf.environment
    gm_d = diff⁺
    up_d = diff⁺.edmf.updraft
    en_d = diff⁺.edmf.environment
    # Charlie is the use of state⁺ here consistent for gm.ρ, up[i].ρa ?
    if bctype == 1 # bottom
        area_en  = 1 - sum([up[i].ρa for i in 1:N])/gm.ρ
        # YAIR - I need to pass the SurfaceModel into BC and into env_surface_covariances
        # tke, e_cv ,q_tot_cv ,e_q_tot_cv = env_surface_covariances(m, edmf.surface, edmf, gm)
        tke = FT(0)
        e_cv = FT(0)
        q_tot_cv = FT(0)
        e_q_tot_cv = FT(0)
        en_d.∇tke = SVector(0,0,gm.ρ * area_en * tke)
        en_d.∇e_cv = SVector(0,0,gm.ρ * area_en * e_cv)
        en_d.∇q_tot_cv = SVector(0,0,gm.ρ * area_en * q_tot_cv)
        en_d.∇e_q_tot_cv = SVector(0,0,gm.ρ * area_en * e_q_tot_cv)
    elseif bctype == 2 # top
        # for now zero flux at the top
        en_d.∇tke = -n⁻ * FT(0)
        en_d.∇e_cv = -n⁻ * FT(0)
        en_d.∇q_tot_cv = -n⁻ * FT(0)
        en_d.∇e_q_tot_cv = -n⁻ * FT(0)
    end
end;
