#=
Full testing pipeline for the implementation of model in Abdul-Razzal and Ghan (2000).

Isabella Dula and Shevali Kadakia


Test classifications:
    --Verfication (VER): ensures that function output has consistent output, no matter inputted values (i.e.,
    verifies that the functions are doing what we want them to)
    --Validation (VAL): checks functions against model data in Abdul-Razzak and Ghan (2000) (i.e., validates 
    the functions outputs against published results)

Dimension (DIM):
    --Tests are done with multi-dimensional inputs: 
    --0: Only one mode and one component (e.g., coarse sea salt) 
    --1: Considering multiple modes over one component (e.g., accumulation mode and coarse mode sea salt)
    --2: Considering multiple modes with multiple components (e.g., accumulation and coarse mode for sea 
    salt and dust)

Modes and Components Considered
    --This testing pipeline uses aerosol data from Porter and Clarke (1997) to provide real-world inputs into 
    the functions
    --Modes: Accumulation (ACC) and coarse (COA)
    --Components: Sea Salt (SS) and Dust (DUS)
=#

# IMPORTS:
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

# Vars:
# General: 
T = 273.15 # temperature, assuming STP (K) 
N = 100000000 # aerosol particle density (particles/m^3)
V = 5.5 # updraft velocity (m/s)
R = 8.314462618 # gas constant ((kg m^2) / (s^2 K mol))
avocado = 6.02214076 × 10^23 # avogadro's number 
tau = 1              #   |
Alpha = 1            #   | 
G = 1        #diffusion   }INCOMPLETE
Gamma  = 1           #   |

# Water Properties:
rho_w = 1000 # water density  (kg/m^3)
M_w = 0.01801528 # water molecular mass (kg/mol)
N_w = 0.022414 # water molar density (m^3/mol)

# Sea Salt accumulation and coarse modes: 
OC_SS = 0.9 # osmotic coefficient
M_SS = 0.058443 # sea salt molar mass; kg/mol
rho_SS = 2170 # sea salt density; kg/m^3
nu_SS = 2 # Sea salt dissociation                           TODO
epsilon_SS = 1 # mass fraction                              TODO
r_SS = 1 # mass mixing ratip                                TODO
a_SS_ACC = 0.000000243 # mean particle radius (m)
a_SS_COA = 0.0000015 # mean particle radius (m)
sigma_SS_ACC = 0.0000014 # mean particle stdev (m)
sigma_SS_COA = 0.0000021 # mean particle stdev(m)

#------------------------------------------------------------------------------------------------------
# Test No. 1
# VER; All tests done; DIM 0; ACC, SS
#------------------------------------------------------------------------------------------------------

@testset "hygroscopicity_test" begin

    # assumptions: standard conditions
    # parameters
    osmotic_coefficient = [OC_SS] # no units
    temperature = [T] # K
    aerosol_density = [rho_SS] # kg/m^3
    aerosol_molecular_mass = [M_SS] # kg/mol
    aerosol_particle_density = [N] # 1/cm^3
    water_density = [rho_w] # kg/m^3
    water_molar_density = [N_w] # m^3/mol
    water_molecular_mass = [M_w] # kg/mol

    # hand calculated value
    updraft_velocity = [V]

    avogadro =  avocado

    h = (updraft_velocity .* osmotic_coefficient 
        .* (aerosol_particle_density .* 1./avogadro 
        .* aerosol_molecular_mass)./(1./water_molar_density
        .* 1./1000 .* water_molecular_mass) .* water_molecular_mass
        .* aerosol_particle_density) ./ (aerosol_molecular_mass
        .* water_density)

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
    mass_mixing_ratio = [r_SS]
    dissociation = [nu_SS]
    osmotic_coefficient = [OC_SS]
    mass_fraction = [epsilon_SS]
    aerosol_molecular_mass = [M_SS]
    aerosol_density = [rho_SS]

    water_molecular_mass = [M_w] # kg/mol
    water_density = [rho_w] # kg/m^3

    B_bar = zeros(3)
    
    for i in range(size(mass_mixing_ratio, 2))
        B_bar[i] = ((water_molecular_mass[i])/(water_density[i]))*(sum(mass_mixing_ratio[i].*dissociation[i].*epsilon_SS[i].*(1./aerosol_molecular_mass[i])  ) / sum(mass_mixing_ratio[i]./aerosol_density[i]))

    end
    
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density)[i] = B_bar[i]
    end

end

@testset "max_supersat_test" begin
    # parameters inputted into function:
    particle_radius = [a_SS_ACC] # particle mode radius (m)
    particle_radius_stdev = [sigma_SS_ACC] # standard deviation of mode radius (m)
    activation_time = [tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w] # Density of water (kg/m^3)
    temperature = [T] # Temperature (K)
    alpha = [Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N] # Initial particle concentration (1/m^3)
    gamma = [Gamma] # coefficient 

    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
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
    particle_radius = [a_SS_ACC] # particle mode radius (m)
    activation_time = [tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w] # Density of water (kg/m^3)
    temperature = [T] # Temperature (K)
    particle_radius_stdev = [sigma_SS_ACC] # standard deviation of mode radius (m)
    alpha = [Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V] # Updraft velocity (m/s)
    diffusion = [G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N] # Initial particle concentration (1/m^3)
    gamma = [Gamma] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)
    
    # Final calculation:
    DRSAP = particle_radius.*((S_mi)./(S_max)).^(2/3)
    
    # Running test:
    @test smallactpartdryrad(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = DRSAP
    
end
        
@testset "total_N_Act_test" begin
    # Input parameters
    particle_radius = [a_SS_ACC] # particle mode radius (m)
    activation_time = [tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w] # Density of water (kg/m^3)
    temperature = [T] # Temperature (K)
    particle_radius_stdev = [sigma_SS_ACC] # standard deviation of mode radius (m)
    alpha = [Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V] # Updraft velocity (m/s)
    diffusion = [G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N] # Initial particle concentration (1/m^3)
    gamma = [Gamma] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    u = ((2*log.(S_min/S_max))./(3.*(2.^.5).*log.(particle_radius_stdev)))
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 2
# VER; All tests done; DIM 0; COA, SS
#------------------------------------------------------------------------------------------------------
@testset "hygroscopicity_test" begin

    # assumptions: standard conditions
    # parameters
    osmotic_coefficient = [OC_SS] # no units
    temperature = [T] # K
    aerosol_density = [rho_SS] # kg/m^3
    aerosol_molecular_mass = [M_SS] # kg/mol
    aerosol_particle_density = [N] # 1/cm^3
    water_density = [rho_w] # kg/m^3
    water_molar_density = [N_w] # m^3/mol
    water_molecular_mass = [M_w] # kg/mol

    # hand calculated value
    updraft_velocity = [V]

    avogadro =  avocado

    h = (updraft_velocity .* osmotic_coefficient 
        .* (aerosol_particle_density .* 1./avogadro 
        .* aerosol_molecular_mass)./(1./water_molar_density
        .* 1./1000 .* water_molecular_mass) .* water_molecular_mass
        .* aerosol_particle_density) ./ (aerosol_molecular_mass
        .* water_density)

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
    mass_mixing_ratio = [r_SS]
    dissociation = [nu_SS]
    osmotic_coefficient = [OC_SS]
    mass_fraction = [epsilon_SS]
    aerosol_molecular_mass = [M_SS]
    aerosol_density = [rho_SS]

    water_molecular_mass = [M_w] # kg/mol
    water_density = [rho_w] # kg/m^3

    B_bar = zeros(3)
    
    for i in range(size(mass_mixing_ratio, 2))
        B_bar[i] = ((water_molecular_mass[i])/(water_density[i]))*(sum(mass_mixing_ratio[i].*dissociation[i].*epsilon_SS[i].*(1./aerosol_molecular_mass[i])  ) / sum(mass_mixing_ratio[i]./aerosol_density[i]))

    end
    
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density)[i] = B_bar[i]
    end

end

@testset "max_supersat_test" begin
    # parameters inputted into function:
    particle_radius = [a_SS_COA] # particle mode radius (m)
    particle_radius_stdev = [sigma_SS_COA] # standard deviation of mode radius (m)
    activation_time = [tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w] # Density of water (kg/m^3)
    temperature = [T] # Temperature (K)
    alpha = [Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V] # Updraft velocity (m/s)
    diffusion = [1] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N] # Initial particle concentration (1/m^3)
    gamma = [Gamma] # coefficient 

    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
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
    particle_radius = [a_SS_COA] # particle mode radius (m)
    activation_time = [tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w] # Density of water (kg/m^3)
    temperature = [T] # Temperature (K)
    particle_radius_stdev = [sigma_SS_COA] # standard deviation of mode radius (m)
    alpha = [Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V] # Updraft velocity (m/s)
    diffusion = [G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N] # Initial particle concentration (1/m^3)
    gamma = [Gamma] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)
    
    # Final calculation:
    DRSAP = particle_radius.*((S_mi)./(S_max)).^(2/3)
    
    # Running test:
    @test smallactpartdryrad(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = DRSAP
    
end
        
@testset "total_N_Act_test" begin
    # Input parameters
    particle_radius = [a_SS_COA] # particle mode radius (m)
    activation_time = [tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w] # Density of water (kg/m^3)
    temperature = [T] # Temperature (K)
    particle_radius_stdev = [sigma_SS_COA] # standard deviation of mode radius (m)
    alpha = [Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V] # Updraft velocity (m/s)
    diffusion = [G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N] # Initial particle concentration (1/m^3)
    gamma = [Gamma] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    u = ((2*log.(S_min/S_max))./(3.*(2.^.5).*log.(particle_radius_stdev)))
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 3
# VER; All tests done; DIM 1; ACC & COA, SS
#------------------------------------------------------------------------------------------------------
@testset "hygroscopicity_test" begin

    # assumptions: standard conditions
    # parameters
    osmotic_coefficient = [OC_SS, OC_SS # no units
    temperature = [T, T] # K
    aerosol_density = [rho_SS, rho_w] # kg/m^3
    aerosol_molecular_mass = [M_SS, M_SS] # kg/mol
    aerosol_particle_density = [N, N] # 1/cm^3
    water_density = [rho_w, rho_w] # kg/m^3
    water_molar_density = [N_w, N_w] # m^3/mol
    water_molecular_mass = [M_w, M_w] # kg/mol

    # hand calculated value
    updraft_velocity = [V, V]

    avogadro =  avocado

    h = (updraft_velocity .* osmotic_coefficient 
        .* (aerosol_particle_density .* 1./avogadro 
        .* aerosol_molecular_mass)./(1./water_molar_density
        .* 1./1000 .* water_molecular_mass) .* water_molecular_mass
        .* aerosol_particle_density) ./ (aerosol_molecular_mass
        .* water_density)

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
    mass_mixing_ratio = [r_SS, r_SS]
    dissociation = [nu_SS, nu_SS]
    osmotic_coefficient = [OC_SS, OC_SS]
    mass_fraction = [epsilon_SS, epsilon_SS]
    aerosol_molecular_mass = [M_SS, M_SS]
    aerosol_density = [rho_SS, rho_SS]

    water_molecular_mass = [M_w, M_w] # kg/mol
    water_density = [rho_w, rho_w] # kg/m^3

    B_bar = zeros(3)
    
    for i in range(size(mass_mixing_ratio, 2))
        B_bar[i] = ((water_molecular_mass[i])/(water_density[i]))*(sum(mass_mixing_ratio[i].*dissociation[i].*epsilon_SS[i].*(1./aerosol_molecular_mass[i])  ) / sum(mass_mixing_ratio[i]./aerosol_density[i]))

    end
    
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density)[i] = B_bar[i]
    end

end

@testset "max_supersat_test" begin
    # parameters inputted into function:
    particle_radius = [a_SS_ACC, a_SS_COA] # particle mode radius (m)
    particle_radius_stdev = [a_SS_ACC, sigma_SS_COA] # standard deviation of mode radius (m)
    activation_time = [tau, tau] # time of activation (s)                                                  
    water_molecular_mass = [M_, M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w, rho_w] # Density of water (kg/m^3)
    temperature = [T, T] # Temperature (K)
    alpha = [Alpha, Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V, V] # Updraft velocity (m/s)
    diffusion = [G, G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N, N] # Initial particle concentration (1/m^3)
    gamma = [Gamma, Gamma] # coefficient 

    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
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
    particle_radius = [a_SS_ACC, a_SS_COA] # particle mode radius (m)
    activation_time = [tau, tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w, M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w, rho_w] # Density of water (kg/m^3)
    temperature = [T, T] # Temperature (K)
    particle_radius_stdev = [sigma_SS_ACC, sigma_SS_COA] # standard deviation of mode radius (m)
    alpha = [Alpha, Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V, V] # Updraft velocity (m/s)
    diffusion = [G, G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N, N] # Initial particle concentration (1/m^3)
    gamma = [Gamma, Gamma] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)
    
    # Final calculation:
    DRSAP = particle_radius.*((S_mi)./(S_max)).^(2/3)
    
    # Running test:
    @test smallactpartdryrad(particle_radius, particle_radius_stdev, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = DRSAP
    
end
        
@testset "total_N_Act_test" begin
    # Input parameters
    particle_radius = [a_SS_ACC, a_SS_COA] # particle mode radius (m)
    activation_time = [tau, tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w, M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w, rho_w] # Density of water (kg/m^3)
    temperature = [T, T] # Temperature (K)
    particle_radius_stdev = [sigma_SS_ACC, sigma_SS_COA] # standard deviation of mode radius (m)
    alpha = [Alpha, Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V, V] # Updraft velocity (m/s)
    diffusion = [G, G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [N, N] # Initial particle concentration (1/m^3)
    gamma = [Gamma, Gamma] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    u = ((2*log.(S_min/S_max))./(3.*(2.^.5).*log.(particle_radius_stdev)))
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = totN

end


#------------------------------------------------------------------------------------------------------
# Test No. 4
# VER; All tests done; DIM 2; ACC & COA, SS & DUS
#------------------------------------------------------------------------------------------------------

# TODO: DETERMINE PARAMETERS

#------------------------------------------------------------------------------------------------------
# Test No. 5
# VAL; ONLY total_N_Act_test; DIM 0; 0 CHECK

# ~~~|This test section checks that, if given an initial input of 0 particle density, the number of 
# ~~~|activated particles is 0. 

#------------------------------------------------------------------------------------------------------
@testset "total_N_Act_test" begin
    # Input parameters
    particle_radius = [a_SS_ACC] # particle mode radius (m)
    activation_time = [tau] # time of activation (s)                                                  
    water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
    water_density  = [rho_w] # Density of water (kg/m^3)
    temperature = [T] # Temperature (K)
    particle_radius_stdev = [sigma_SS_ACC] # standard deviation of mode radius (m)
    alpha = [Alpha] # Coefficient in superaturation balance equation       
    updraft_velocity = [V] # Updraft velocity (m/s)
    diffusion = [G] # Diffusion of heat and moisture for particles 
    aerosol_particle_density = [0] # Initial particle concentration (1/m^3)
    gamma = [Gamma] # coefficient 
    
    # Internal calculations:
    B_bar = mean_hygrosopicity(mass_mixing_ratio, disassociation, osmotic_coefficient, mass_fraction, aerosol_molecular_mass, aerosol_density) # calculated in earlier function    ------ INCOMPLETE-------
    A = (2.*activation_time.*water_molecular_mass)./(water_density .*R.*temperature) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)./(B_bar).^(.5)).*((A)./(3.*particle_radius)).^(3/2) # Minimum supersaturation
    S_max = maxsupersat(particle_radius, particle_radius_stdev activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)

    u = ((2*log.(S_min/S_max))./(3.*(2.^.5).*log.(particle_radius_stdev)))
    # Final Calculation: 
    totN = sum(aerosol_particle_density.*.5.*(1-erf.(u)))

    # Compare results:
    @test total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma) = 0
end

#------------------------------------------------------------------------------------------------------
# Test No. 6
# VAL; ONLY total_N_Act_test; DIM 0; Paper validation

# ~~~|This test section checks compares outputs to results give in the 
# ~~~|Abdul-Razzal and Ghan paper to validate the accuracy of the 
# ~~~|functions implemented. 

#------------------------------------------------------------------------------------------------------
# Paper parameters/constants (AG=data given in Abdul-Razzak & Ghan (2000) paper)
T_AG = 294 # K 
V_AG = 0.5 # m/s
a_AG = 5*10^(-8)
sigma_AG = 2
nu_AG = 3
M_AG = 132 # kg/mol
rho_AG = 1770 # kg/m^3
N_AG = 100000000 # 1/m^3

# 6a: Validating initial particle density versus final activation number. 

@testset "Validate_initial_N" begin
    # test paramters
    Ns = [810.7472258, 3555.037372, 2936.468114. 2515.271387. 1778.226562. 1166.163519]
    totNs = [0.613108201, 0.507426454, 0.525448148, 0.542970165, 0.567620911, 0.594486894]
    
    


    for i in range(size(Ns))
        # Input parameters  TODO: figure out remaining parameters
        particle_radius = [a_AG] # particle mode radius (m)
        activation_time = [tau] # time of activation (s)                                                  
        water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
        water_density  = [rho_w] # Density of water (kg/m^3)
        temperature = [T_AG] # Temperature (K)
        particle_radius_stdev = [sigma_AG] # standard deviation of mode radius (m)
        alpha = [Alpha] # Coefficient in superaturation balance equation       
        updraft_velocity = [V_AG] # Updraft velocity (m/s)
        diffusion = [G] # Diffusion of heat and moisture for particles 
        aerosol_particle_density = [Ns[i]] # Initial particle concentration (1/m^3)
        gamma = [Gamma] # coefficient 
         
        totN = totNs[i]

        func_totN = total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)
        


        # Compare results:
        @test  ((totN-functotN)/totN)<.1 # checks if we are within a certain percent error of paper results

    end


end

# 6b: Validating initial updraft velocity versus final activation number

@testset "Validate_initial_updraft_velocity" begin
    # test paramters
    Vs = [0.200895541, 0.350392221, 0.048877492, 0.02335161, 3.155076552, 0.781752551, 0.016214404]
    totNs = [0.539319421, 0.647329779, 0.318536481, 0.22662361, 0.931693526, 0.785977761, 0.178818583]
    
    for i in range(size(Vs))
        # Input parameters  TODO: figure out remaining parameters
        particle_radius = [a_AG] # particle mode radius (m)
        activation_time = [tau] # time of activation (s)                                                  
        water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
        water_density  = [rho_w] # Density of water (kg/m^3)
        temperature = [T_AG] # Temperature (K)
        particle_radius_stdev = [sigma_AG] # standard deviation of mode radius (m)
        alpha = [Alpha] # Coefficient in superaturation balance equation       
        updraft_velocity = [Vs[i]] # Updraft velocity (m/s)
        diffusion = [G] # Diffusion of heat and moisture for particles 
        aerosol_particle_density = [N_AG] # Initial particle concentration (1/m^3)
        gamma = [Gamma] # coefficient 
        
        totN = totNs[i]

        func_totN = total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)


        # Compare results:
        @test  ((totN-functotN)/totN)<.1 # checks if we are within a certain percent error of paper results

    end


end

# 6c: Validating initial mode radius versus final activation number 

@testset "Validate_initial_mode_radius"  begin
    # test paramters
    as = [0.121074949, 0.246347032, 0.365327666, 0.015725944, 0.03836725, 0.089315933, 0.029013032]
    totNs = [0.536374647, 0.319953771, 0.193993366, 0.73419934, 0.695886073, 0.615805014, 0.707858178]

        for i in range(size(Vs))
            # Input parameters  TODO: figure out remaining parameters
            particle_radius = [as[i]] # particle mode radius (m)
            activation_time = [tau] # time of activation (s)                                                  
            water_molecular_mass = [M_w] # Molecular weight of water (kg/mol)
            water_density  = [rho_w] # Density of water (kg/m^3)
            temperature = [T_AG] # Temperature (K)
            particle_radius_stdev = [sigma_AG] # standard deviation of mode radius (m)
            alpha = [Alpha] # Coefficient in superaturation balance equation       
            updraft_velocity = [V_AG] # Updraft velocity (m/s)
            diffusion = [G] # Diffusion of heat and moisture for particles 
            aerosol_particle_density = [N_AG] # Initial particle concentration (1/m^3)
            gamma = [Gamma] # coefficient 

            totN = totNs[i]

            func_totN = total_N_Act(particle_radius, activation_time, water_molecular_mass, water_density , R, temperature, B_i_bar, particle_radius_stdev, alpha, updraft_velocity, diffusion, aerosol_particle_density, gamma)
            


            # Compare results:
            @test  ((totN-functotN)/totN)<.1 # checks if we are within a certain percent error of paper results

        end
end


# TODO: Implement more detailed validation tests
# TODO: Figure out missing constants