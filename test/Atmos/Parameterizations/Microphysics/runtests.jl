Pkg.add("SpecialFunctions")
using Test
using ClimateMachine.Microphysics_0M
using ClimateMachine.Microphysics
using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics
using CLIMAParameters.Atmos.Microphysics_0M

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end

struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
    liquid::L
    ice::I
    rain::R
    snow::S
end

struct EarthParamSet{M} <: AbstractEarthParameterSet
    microphys_param_set::M
end

microphys_param_set = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)

prs = EarthParamSet(microphys_param_set)
liq_prs = prs.microphys_param_set.liquid
ice_prs = prs.microphys_param_set.ice
rai_prs = prs.microphys_param_set.rain
sno_prs = prs.microphys_param_set.snow

@testset "τ_relax" begin

    @test τ_relax(liq_prs) ≈ 10
    @test τ_relax(ice_prs) ≈ 10

end

@testset "0M_microphysics" begin

    _τ_precip = τ_precip(prs)
    _qc_0 = qc_0(prs)
    _S_0 = S_0(prs)

    q_vap_sat = 10e-3
    qc = 3e-3
    q_tot = 13e-3
    frac = [0.0, 0.5, 1.0]

    # no rain if no cloud
    q = PhasePartition(q_tot, 0.0, 0.0)
    @test remove_precipitation(prs, q) ≈ 0
    @test remove_precipitation(prs, q, q_vap_sat) ≈ 0

    # rain based on qc threshold
    for lf in frac
        q_liq = qc * lf
        q_ice = (1 - lf) * qc

        q = PhasePartition(q_tot, q_liq, q_ice)

        @test remove_precipitation(prs, q) ≈
              -max(0, q_liq + q_ice - _qc_0) / _τ_precip
    end

    # rain based on supersaturation threshold
    for lf in frac
        q_liq = qc * lf
        q_ice = (1 - lf) * qc

        q = PhasePartition(q_tot, q_liq, q_ice)

        @test remove_precipitation(prs, q, q_vap_sat) ≈
              -max(0, q_liq + q_ice - _S_0 * q_vap_sat) / _τ_precip
    end
end

@testset "RainFallSpeed" begin
    # eq. 5d in [Grabowski1996](@cite)
    function terminal_velocity_empir(
        q_rai::FT,
        q_tot::FT,
        ρ::FT,
        ρ_air_ground::FT,
    ) where {FT <: Real}
        rr = q_rai / (1 - q_tot)
        vel = FT(14.34) * ρ_air_ground^FT(0.5) * ρ^-FT(0.3654) * rr^FT(0.1346)
        return vel
    end

    # some example values
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    ρ_air, q_tot, ρ_air_ground = 1.2, 20 * 1e-3, 1.22

    for q_rai in q_rain_range
        @test terminal_velocity(prs, rai_prs, ρ_air, q_rai) ≈
              terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground) atol =
            0.2 * terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground)

    end
end

@testset "CloudLiquidCondEvap" begin

    q_liq_sat = 5e-3
    frac = [0.0, 0.5, 1.0, 1.5]

    _τ_cond_evap = τ_relax(liq_prs)

    for fr in frac
        q_liq = q_liq_sat * fr

        @test conv_q_vap_to_q_liq_ice(
            liq_prs,
            PhasePartition(0.0, q_liq_sat, 0.0),
            PhasePartition(0.0, q_liq, 0.0),
        ) ≈ (1 - fr) * q_liq_sat / _τ_cond_evap
    end
end

@testset "CloudIceCondEvap" begin

    q_ice_sat = 2e-3
    frac = [0.0, 0.5, 1.0, 1.5]

    _τ_cond_evap = τ_relax(ice_prs)

    for fr in frac
        q_ice = q_ice_sat * fr

        @test conv_q_vap_to_q_liq_ice(
            ice_prs,
            PhasePartition(0.0, 0.0, q_ice_sat),
            PhasePartition(0.0, 0.0, q_ice),
        ) ≈ (1 - fr) * q_ice_sat / _τ_cond_evap
    end
end

@testset "RainAutoconversion" begin

    _q_liq_threshold = q_liq_threshold(rai_prs)
    _τ_acnv = τ_acnv(rai_prs)

    q_liq_small = 0.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai(rai_prs, q_liq_small) == 0.0

    q_liq_big = 1.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai(rai_prs, q_liq_big) ==
          0.5 * _q_liq_threshold / _τ_acnv
end

@testset "SnowAutoconversion" begin

    ρ = 1.0

    # above freezing temperatures -> no snow
    q = PhasePartition(15e-3, 2e-3, 1e-3)
    T = 273.15 + 30
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) == 0.0

    # no ice -> no snow
    q = PhasePartition(15e-3, 2e-3, 0.0)
    T = 273.15 - 30
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) == 0.0

    # no supersaturation -> no snow
    T = 273.15 - 5
    q_sat_ice = q_vap_saturation_generic(prs, T, ρ, Ice())
    q = PhasePartition(q_sat_ice, 2e-3, 3e-3)
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) == 0.0

    # TODO - coudnt find a plot of what it should be from the original paper
    # just chacking if the number stays the same
    T = 273.15 - 10
    q_vap = 1.02 * q_vap_saturation_generic(prs, T, ρ, Ice())
    q_liq = 0.0
    q_ice = 0.03 * q_vap
    q = PhasePartition(q_vap + q_liq + q_ice, q_liq, q_ice)
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) ≈ 1.8512022335645584e-9

end

@testset "RainLiquidAccretion" begin

    # eq. 5b in [Grabowski1996](@cite)
    function accretion_empir(q_rai::FT, q_liq::FT, q_tot::FT) where {FT <: Real}
        rr = q_rai / (FT(1) - q_tot)
        rl = q_liq / (FT(1) - q_tot)
        return FT(2.2) * rl * rr^FT(7 / 8)
    end

    # some example values
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    ρ_air, q_liq, q_tot = 1.2, 5e-4, 20e-3

    for q_rai in q_rain_range
        @test accretion(prs, liq_prs, rai_prs, q_liq, q_rai, ρ_air) ≈
              accretion_empir(q_rai, q_liq, q_tot) atol =
            (0.1 * accretion_empir(q_rai, q_liq, q_tot))
    end
end

@testset "Accretion" begin
    # TODO - coudnt find a plot of what it should be from the original paper
    # just chacking if the number stays the same

    # some example values
    ρ = 1.2
    q_tot = 20e-3
    q_ice = 5e-4
    q_sno = 5e-4
    q_liq = 5e-4
    q_rai = 5e-4

    @test accretion(prs, liq_prs, rai_prs, q_liq, q_rai, ρ) ≈
          1.4150106417043544e-6
    @test accretion(prs, ice_prs, sno_prs, q_ice, q_sno, ρ) ≈
          2.453070979562392e-7
    @test accretion(prs, liq_prs, sno_prs, q_liq, q_sno, ρ) ≈
          2.453070979562392e-7
    @test accretion(prs, ice_prs, rai_prs, q_ice, q_rai, ρ) ≈
          1.768763302130443e-6

    @test accretion_rain_sink(prs, ice_prs, rai_prs, q_ice, q_rai, ρ) ≈
          3.085229094251214e-5

    @test accretion_snow_rain(prs, sno_prs, rai_prs, q_sno, q_rai, ρ) ≈
          2.1705865794293408e-4
    @test accretion_snow_rain(prs, rai_prs, sno_prs, q_rai, q_sno, ρ) ≈
          6.0118801860768854e-5
end

@testset "RainEvaporation" begin

    # eq. 5c in [Grabowski1996](@cite)
    function rain_evap_empir(
        prs::AbstractParameterSet,
        q_rai::FT,
        q::PhasePartition,
        T::FT,
        p::FT,
        ρ::FT,
    ) where {FT <: Real}

        q_sat = q_vap_saturation_generic(prs, T, ρ, Liquid())
        q_vap = q.tot - q.liq
        rr = q_rai / (1 - q.tot)
        rv_sat = q_sat / (1 - q.tot)
        S = q_vap / q_sat - 1

        ag, bg = FT(5.4 * 1e2), FT(2.55 * 1e5)
        G = FT(1) / (ag + bg / p / rv_sat) / ρ

        av, bv = FT(1.6), FT(124.9)
        F =
            av * (ρ / FT(1e3))^FT(0.525) * rr^FT(0.525) +
            bv * (ρ / FT(1e3))^FT(0.7296) * rr^FT(0.7296)

        return 1 / (1 - q.tot) * S * F * G
    end

    # example values
    T, p = 273.15 + 15, 90000.0
    ϵ = 1.0 / molmass_ratio(prs)
    p_sat = saturation_vapor_pressure(prs, T, Liquid())
    q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    q_tot = 15e-3
    q_vap = 0.15 * q_sat
    q_ice = 0.0
    q_liq = q_tot - q_vap - q_ice
    q = PhasePartition(q_tot, q_liq, q_ice)
    R = gas_constant_air(prs, q)
    ρ = p / R / T

    for q_rai in q_rain_range
        @test evaporation_sublimation(prs, rai_prs, q, q_rai, ρ, T) ≈
              rain_evap_empir(prs, q_rai, q, T, p, ρ) atol =
            -0.5 * rain_evap_empir(prs, q_rai, q, T, p, ρ)
    end

    # no condensational growth for rain
    T, p = 273.15 + 15, 90000.0
    ϵ = 1.0 / molmass_ratio(prs)
    p_sat = saturation_vapor_pressure(prs, T, Liquid())
    q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))
    q_rai = 1e-4
    q_tot = 15e-3
    q_vap = 1.15 * q_sat
    q_ice = 0.0
    q_liq = q_tot - q_vap - q_ice
    q = PhasePartition(q_tot, q_liq, q_ice)
    R = gas_constant_air(prs, q)
    ρ = p / R / T

    @test evaporation_sublimation(prs, rai_prs, q, q_rai, ρ, T) ≈ 0.0

end

@testset "SnowSublimation" begin
    # TODO - coudnt find a plot of what it should be from the original paper
    # just chacking if the number stays the same

    cnt = 0
    ref_val = [
        -1.9756907119482267e-7,
        1.9751292385808357e-7,
        -1.6641552112891826e-7,
        1.663814937710236e-7,
    ]
    # some example values
    for T in [273.15 + 2, 273.15 - 2]
        p = 90000.0
        ϵ = 1.0 / molmass_ratio(prs)
        p_sat = saturation_vapor_pressure(prs, T, Ice())
        q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))

        for eps in [0.95, 1.05]
            cnt += 1

            q_tot = eps * q_sat
            q_ice = 0.0
            q_liq = 0.0
            q = PhasePartition(q_tot, q_liq, q_ice)

            q_sno = 1e-4

            R = gas_constant_air(prs, q)
            ρ = p / R / T

            @test evaporation_sublimation(prs, sno_prs, q, q_sno, ρ, T) ≈
                  ref_val[cnt]

        end
    end
end

@testset "SnowMelt" begin

    # TODO - find a good reference to compare with
    T = 273.15 + 2
    ρ = 1.2
    q_sno = 1e-4
    @test snow_melt(prs, sno_prs, q_sno, ρ, T) ≈ 9.518235437405256e-6

    # no snow -> no snow melt
    T = 273.15 + 2
    ρ = 1.2
    q_sno = 0.0
    @test snow_melt(prs, sno_prs, q_sno, ρ, T) ≈ 0

    # T < T_freeze -> no snow melt
    T = 273.15 - 2
    ρ = 1.2
    q_sno = 1e-4
    @test snow_melt(prs, sno_prs, q_sno, ρ, T) ≈ 0

end



#-----------------------------------------------------------------------------------------------#



@testset "hygroscopicity_test" begin

    # assumptions: standard conditions
    # parameters
    osmotic_coefficient = 3 # no units
    temperature = 298 # K
    aerosol_density = 1770 # kg/m^3
    aerosol_molecular_mass = 0.132 # kg/mol
    aerosol_particle_density = 100000000# 1/cm^3
    water_density = 1000 # kg/m^3
    water_molar_density = 0.022414 # m^3/mol
    water_molecular_mass = 0.01801528 # kg/mol

    # hand calculated value
    updraft_velocity = 3
    R = 8.31446261815324
    avogadro =  6.02214076 × 10^23

    h = (updraft_velocity * osmotic_coefficient 
        * (aerosol_particle_density * 1/avogadro 
        * aerosol_molecular_mass)/(1/water_molar_density
        * 1/1000 * water_molecular_mass) * water_molecular_mass
        * aerosol_particle_density) / (aerosol_molecular_mass
        * water_density)

    @test hygroscopicity(
               osmotic_coefficient,
               temperature,
               aerosol_density,
               aerosol_molecular_mass,
               aerosol_particle_density
               water_density,
               water_molar_density,
               water_molecular_mass,
               ) ≈ h
end 

@testset "Mean Hygroscopicity" begin
    aerosol_component = [1, 2, 3]        
    aerosol_mode_number = [1, 2, 3, 4, 5]
    mass_mixing_ratio = [[1, 2, 3, 4, 5]
                         [0.1, 0.2, 0.3, 0.4, 0.5]
                         [0.01, 0.02, 0.03, 0.04, 0.05]]
    dissociation = [[1, 2, 3, 4, 5]
                      [0.1, 0.2, 0.3, 0.4, 0.5]
                      [0.01, 0.02, 0.03, 0.04, 0.05]]
    osmotic_coefficient = [[1, 2, 3, 4, 5]
           [0.1, 0.2, 0.3, 0.4, 0.5]
           [0.01, 0.02, 0.03, 0.04, 0.05]]
    mass_fraction = [[1, 2, 3, 4, 5]
               [0.1, 0.2, 0.3, 0.4, 0.5]
               [0.01, 0.02, 0.03, 0.04, 0.05]]
    aerosol_molecular_mass = [[1, 2, 3, 4, 5]
                        [0.1, 0.2, 0.3, 0.4, 0.5]
                        [0.01, 0.02, 0.03, 0.04, 0.05]]
    aerosol_density = [[1, 2, 3, 4, 5]
                       [0.1, 0.2, 0.3, 0.4, 0.5]
                       [0.01, 0.02, 0.03, 0.04, 0.05]]
    add_top = 0
    add_bottom = 0
    water_molecular_mass = 0.01801528 # kg/mol
    water_density = 1000 # kg/m^3
    m_h = zeros(3)
    top_values = mass_mixing_ratio .* dissociation .* osmotic_coefficient .* mass_fraction .*aerosol_molecular_mass
    top_values *= water_molecular_mass
    bottom_values = mass_mixing_ratio ./ aerosol_density
    bottom_values *= water_density

    for i in 1:length(aerosol_component)
        m_h[i] = sum(top_values[aerosol_component]) * water_molecular_mass / (sum(bottom_values) 
        end
        m_h[i] = add_top / bottom_values
    return m_h
    end

    @test mean_hygroscopicity(aerosol_mode_number[1], 
                              aerosol_component), 
                              mass_mixing_ratio,
                              disassociation,
                              osmotic_coefficient,
                              mass_fraction,
                              aerosol_molecular_mass,
                              ) ≈ m_h[1]
    @test mean_hygroscopicity(aerosol_component[2], 
                              aerosol_mode_number), 
                              mass_mixing_ratio,
                              disassociation,
                              osmotic_coefficient,
                              mass_fraction,
                              aerosol_molecular_mass,
                              ) ≈ m_h[2]
    @test mean_hygroscopicity(aerosol_component[3], 
                              aerosol_mode_number), 
                              mass_mixing_ratio,
                              disassociation,
                              osmotic_coefficient,
                              mass_fraction,
                              aerosol_molecular_mass,
                              ) ≈ m_h[3]
end


@testset "max_supersat_test" begin
    # parameters inputted into function:
    particle_radius = [5*(10^(-8))] # particle mode radius (m)
    particle_radius_stdev = [2] # standard deviation of mode radius (m)
    activation_time = [1] # time of activation (s)                                                  
    water_molecular_mass = [0.01801528] # Molecular weight of water (kg/mol)
    water_density  = [1000] # Density of water (kg/m^3)
    R = [8.31446261815324] # Gas constant (kg*m^2/s^2*K*mol)
    temperature = [273.15] # Temperature (K)
    alpha = [1] # Coefficient in superaturation balance equation       
    updraft_velocity = [1] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [100000000] # Initial particle concentration (1/m^3)
    gamma = [1] # coefficient 

    # Internal calculations:
    B_bar = mean_hygrosopicity() # calculated in earlier function    ------ INCOMPLETE-------
    f = 0.5 .* exp(2.5*(log.(particle_radius_stdev)).^2) # function of particle_radius_stdev (check units)
    g = 1 .+ 0.25 .* log.(particle_radius_stdev) # function of particle_radius_stdev (log(m))
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    zeta = ((2.*A)./(3)).*((alpha.*updraft_velocity)/(diffusion)).^(.5) # dependent parameter 
    eta = (  ((alpha.*updraft_velocity)./(diffusion)).^(3/2)  ./    (2*pi.*water_density .*gamma.*aerosol_particle_density)   )    # dependent parameter

    
    # Final value for maximum supersaturation:
    MS = sum(((1)./(((S_min).^2) * (    f_i.*((zeta./eta).^(3/2))     +    g_i.*(((S_min.^2)./(eta+3.*zeta)).^(3/4))    ) )))

    # Comaparing calculated MS value to function output: 
    @test maxsupersat(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = MS
end

@testset "smallactpartdryrad_test" begin
    # Parameter inputs:
    particle_radius = [5*(10^(-8))] # particle mode radius (m)
    activation_time = [1] # time of activation (s)                                                  
    water_molecular_mass = [0.01801528] # Molecular weight of water (kg/mol)
    water_density  = [1000] # Density of water (kg/m^3)
    R = [8.31446261815324] # Gas constant (kg)
    temperature = [273.15] # Temperature (K)
    particle_radius_stdev = [2] # standard deviation of mode radius (m)
    alpha = [1] # Coefficient in superaturation balance equation       
    updraft_velocity = [1] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [100000000] # Initial particle concentration (1/m^3)
    gamma = [1] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity() # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)
    
    # Final calculation:
    DRSAP = particle_radius.*((S_mi)./(S_max)).^(2/3)
    
    # Running test:
    @test smallactpartdryrad(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = DRSAP
    
    end
        
    Pkg.add("SpecialFunctions")

@testset "total_N_Act_test" begin
    # Input parameters
    particle_radius = [5*(10^(-8))] # particle mode radius (m)
    activation_time = [1] # time of activation (s)                                                  
    water_molecular_mass = [0.01801528] # Molecular weight of water (kg/mol)
    water_density  = [1000] # Density of water (kg/m^3)
    R = [8.31446261815324] # Gas constant (kg)
    temperature = [273.15] # Temperature (K)
    particle_radius_stdev = [2] # standard deviation of mode radius (m)
    alpha = [1] # Coefficient in superaturation balance equation       
    updraft_velocity = [1] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [100000000] # Initial particle concentration (1/m^3)
    gamma = [1] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity() # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    u = ((2*log.(S_min/S_max))./(3.*(2.^.5).*log.(particle_radius_stdev)))
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = totN

end

@testset alpha()
    gravity = 9.8
    water_molecular_mass = 0.01801528
    latent_heat_evaporation = 2.26 * 10^-6
    specific_heat_air = 1000
    r = 31446261815324
    t = 273.15
    M_a = 0.058443
    a = (gravity * water_molecular_mass * latent_heat_evaporation) 
        / (specific_heat_air * r * t^2) - gravity * M_a / (r * t)
    @test alpha_sic(T, M_a) ≈ a
end

@testset gamma()
    gravity = 9.8
    water_molecular_mass = 0.01801528
    latent_heat_evaporation = 2.26 * 10^-6
    specific_heat_air = 1000
    r = 31446261815324
    t = 273.15
    M_a = 0.058443
    rho_s = 2170
    g = r * t / (rho_s * water_molecular_mass) + water_molecular_mass 
        * specific_heat_air ^ 2 / (specific_heat_air * water_density * M_a * T)
    @test gamma_sic(t, M_a, rho_s) ≈ g
end

end 