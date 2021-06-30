Pkg.add("SpecialFunctions")
# TODO: Map correct locations to run the test 
using Test
using ClimateMachine.Microphysics_0M
using ClimateMachine.Microphysics
using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics
using CLIMAParameters.Atmos.Microphysics_0M

"""
Isabella Dula and Shevali Kadakia
    
This file has the complete set of tests to verify and validate the parameterization of
the model given in Abdul-Razzak and Ghan (2000). 

The structure of the pipeline is as follows:

--Test classifications:
    --Verfication (VER): ensures that function output has consistent output, no matter inputted values (i.e.,
    verifies that the functions are doing what we want them to)
    --Validation (VAL): checks functions against model data in Abdul-Razzak and Ghan (2000) (i.e., validates 
    the functions outputs against published results)

--Dimension (DIM):
    --Tests are done with multi-dimensional inputs: 
    --0: Only one mode and one component (e.g., coarse sea salt) 
    --1: Considering multiple modes over one component (e.g., accumulation mode and coarse mode sea salt)
    --2: Considering multiple modes with multiple components (e.g., accumulation and coarse mode for sea 
    salt and dust)

--Modes and Components Considered
        --This testing pipeline uses aerosol data from Porter and Clarke (1997) to provide real-world inputs into 
        the functions
        --Modes: Accumulation (ACC) and coarse (COA)
        --Components: Sea Salt (SS) and Dust (DUS)

"""
# CONSTANTS
WTR_ρ = 1000
WTR_MLR_ρ = 0.022414
WTR_MM = 0.01801528
R = 8.31446261815324
AVO = 6.02214076 × 10^23
G = 9.8
LAT_HEAT_EVP = 2.26 * 10^-6
SPC_HEAT_AIR = 1000
TEMP = 273.1 # K (STP)
P = 100000 # Pa (N/m^2) (STP)
K = 2.4*10^(-2) # J/m*s*K
R_v = R/WTR_MM 
D = 2.26 * 10 ^ (-5) # m^2/s


# TEST INPUTS 
P_sat = 100000 # Pa                                  TODO
G = (((LAT_HEAT_EVP/(K*T))*((LAT_HEAT_EVP/(R_v*T))-1))+
       ((R_v*T)/(P_sat*D)))^(-1) # diffusion

# --Sea Salt accumulation and coarse modes:
OC_SS = 0.9 # osmotic coefficient
M_SS = 0.058443 # sea salt molar mass; kg/mol
rho_SS = 2170 # sea salt density; kg/m^3
nu_SS = 2 # Sea salt dissociatio                         TODO
epsilon_SS = 1 # mass fraction                              TODO
r_SS = 1 # mass mixing rati0                               TODO
a_SS_ACC = 0.000000243 # mean particle radius (m)
a_SS_COA = 0.0000015 # mean particle radius (m)
sigma_SS_ACC = 0.0000014 # mean particle stdev (m)
sigma_SS_COA = 0.0000021 # mean particle stdev(m)

#------------------------------------------------------------------------------------------------------
# Test No. 1
# VER; All tests done; DIM 0; ACC, SS
#------------------------------------------------------------------------------------------------------

@testset "Mean_Hygroscopicity_1" begin
    # Parameters for calculations and function input
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    # Internal calculations
    B_bar = zeros(size(mass_mx_rat)[2])
    for i in range(size(mass_mx_rat)[2])
        B_bar[i] = ((WTR_MM)/(WTR_ρ)).*
                   (sum(mass_mx_rat[i].*diss[i].*
                   mass_frac[i].*(1./aero_mm[i])  ) 
                   / sum(mass_mx_rat[i]./aero_ρ[i]))

    end
    
    # Comparison between calculations and function output
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mx_rat, diss, 
                                  osm_coeff, mass_frac,
                                  aero_mm, 
                                  aero_ρ)[i] 
                                  ≈ B_bar[i]
    end
    
    @test size(B_bar) = size([1])

end

@testset "max_supersat_1" begin
    # Parameters for calculations and function input
    part_radius = a_SS_ACC # particle mode radius (m)
    part_radius_stdev = sigma_SS_ACC # standard deviation of mode radius (m)
    act_time = tau # time of activation (s)                                                  
    updft_velo = V # Updraft velocity (m/s)
    diff = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    aero_ρ = rho_SS
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS

    gamma = gamma_sic(M_SS, P_sat) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 
    
    f = 0.5 .* exp(2.5*(log.(part_radius_stdev)).^2) 
    g = 1 .+ 0.25 .* log.(part_radius_stdev) 

    coeff_of_curve = coeff_of_curve(act_time)
    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    zeta = ((2.*coeff_of_curve)./(3)).*((alpha.*updft_velo)/(diff)).^
           (.5) 

    eta = (  ((alpha.*updft_velo)./(diff)).^(3/2)  ./    
          (2*pi.*WTR_ρ .*gamma.*aero_part_ρ)   )    
          

    
    # Final value:
    MS = sum(((1)./(((S_m).^2) * (    f.*((zeta./eta).^(3/2))     
    .+    g.*(((S_m.^2)./(eta+3.*zeta)).^(3/4))    ) )))

    # Comparison between calculations and function output
    @test maxsupersat(part_radius, part_radius_stdev, act_time, 
                      updft_velo, diff, 
                      aero_part_ρ, mass_mx_rat, 
                      diss, 
                      osm_coeff, mass_frac,
                      aero_mm, aero_ρ) 
                      ≈ MS
end
        
@testset "total_N_Act_1" begin
    # Parameters for calculations and function input
    part_radius = a_SS_ACC # particle mode radius (m)
    act_time = tau # time of activation (s)                                                  
    part_radius_stdev = sigma_SS_ACC # standard deviation of mode radius (m)
    updft_velo = V # Updraft velocity (m/s)
    diff = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    gamma = gamma_sic(M_SS, P_sat) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       


    # Internal calculations
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    coeff_of_curve = coeff_of_curve(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diff, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))
    
    # Final value:
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Comparison between calculations and function output
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diff, aero_part_ρ) ≈ totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 2
# VER; All tests done; DIM 0; COA, SS
#------------------------------------------------------------------------------------------------------

@testset "Mean_Hygroscopicity_2" begin
    # Parameters for calculations and function input
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    # Internal calculations
    B_bar = zeros(size(mass_mx_rat)[2])
    for i in range(size(mass_mx_rat)[2])
        B_bar[i] = ((WTR_MM)/(WTR_ρ)).*
                   (sum(mass_mx_rat[i].*diss[i].*
                   epsilon_SS[i].*(1./aero_mm[i])  ) 
                   / sum(mass_mx_rat[i]./aero_ρ[i]))

    end

    # Comparison between calculations and function output
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mx_rat, diss, 
                                  osm_coeff, mass_frac,
                                  aero_mm, 
                                  aero_ρ)[i] 
                                  ≈ B_bar[i]
    end
    @test size(B_bar) = size([1])
end

@testset "max_supersat_2" begin
    # Parameters for calculations and function input
    part_radius = a_SS_COA # particle mode radius (m)
    part_radius_stdev = sigma_SS_COA # standard deviation of mode radius (m)
    act_time = tau # time of activation (s)                                                  
    updft_velo = V # Updraft velocity (m/s)
    diff = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    aero_ρ = rho_SS
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS

    gamma = gamma_sic(M_SS, P_sat) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 
    
    f = 0.5 .* exp(2.5*(log.(part_radius_stdev)).^2) 
    g = 1 .+ 0.25 .* log.(part_radius_stdev) 

    coeff_of_curve = coeff_of_curve(act_time)
    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    zeta = ((2.*coeff_of_curve)./(3)).*((alpha.*updft_velo)/(diff)).^
           (.5) 

    eta = (  ((alpha.*updft_velo)./(diff)).^(3/2)  ./    
          (2*pi.*WTR_ρ .*gamma.*aero_part_ρ)   )    
          

    
    # Final value:
    MS = sum(((1)./(((S_m).^2) * (    f.*((zeta./eta).^(3/2))     
    .+    g.*(((S_m.^2)./(eta+3.*zeta)).^(3/4))    ) )))

    # Comparison between calculations and function output
    @test maxsupersat(part_radius, part_radius_stdev, act_time, 
                      updft_velo, diff, 
                      aero_part_ρ, mass_mx_rat, 
                      diss, 
                      osm_coeff, mass_frac,
                      aero_mm, aero_ρ) 
                      ≈ MS
end
        
@testset "total_N_Act_2" begin
    # Parameters for calculations and function input
    part_radius = a_SS_COA # particle mode radius (m)
    act_time = tau # time of activation (s)                                                  
    part_radius_stdev = sigma_SS_COA # standard deviation of mode radius (m)
    updft_velo = V # Updraft velocity (m/s)
    diff = DIF # Diffusion of heat and moisture for particles 
    aero_part_ρ = N # Initial particle concentration (1/m^3)
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS

    gamma = gamma_sic(M_SS, P_sat) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       


    # Internal calculations
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    coeff_of_curve = coeff_of_curve(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diff, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))

    # Final value:
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Comparison between calculations and function output
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diff, aero_part_ρ) ≈ totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 3
# VER; All tests done; DIM 1; ACC & COA, SS
#------------------------------------------------------------------------------------------------------

@testset "Mean_Hygroscopicity_3" begin
    # Parameters for calculations and function input    
    mass_mx_rat = [r_SS, r_SS]
    diss = [nu_SS, nu_SS]
    osm_coeff = [OC_SS, OC_SS]
    mass_frac = [epsilon_SS, epsilon_SS]
    aero_mm = [M_SS, M_SS]
    aero_ρ = [rho_SS, rho_SS]

    B_bar = zeros(size(mass_mx_rat)[2])
    for i in range(size(mass_mx_rat)[2])
        B_bar[i] = ((WTR_MM)/(WTR_ρ)).*
                   (sum(mass_mx_rat[i].*diss[i].*
                   epsilon_SS[i].*(1./aero_mm[i])  ) 
                   / sum(mass_mx_rat[i]./aero_ρ[i]))

    end
    
    # Comparison between calculations and function output
    for i in size(B_bar)
        @test mean_hygroscopicity(mass_mx_rat, diss, 
                                  osm_coeff, mass_frac,
                                  aero_mm, 
                                  aero_ρ)[i] 
                                  ≈ B_bar[i]
    end
    @test size(B_bar) = size([1, 1])

end

@testset "max_supersat_3" begin
    # Parameters for calculations and function input
    part_radius = [a_SS_ACC, a_SS_COA] 
    part_radius_stdev = [sigma_SS_ACC, sigma_SS_COA] 
    act_time = [tau, tau]                                                 
    updft_velo = [V, V] 
    diff = [DIF, DIF] # Diffusion of heat and moisture for particles 
    aero_part_ρ = [N, N] # Initial particle concentration (1/m^3)
    aero_ρ = [rho_SS, rho_SS]
    mass_mx_rat = [r_SS, r_SS]
    diss = [nu_SS, nu_SS]
    osm_coeff = [OC_SS, OC_SS]
    mass_frac = [epsilon_SS, epsilon_SS]
    aero_mm = [M_SS, M_SS]

    gamma = gamma_sic(M_SS, P_sat) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 
    
    f = 0.5 .* exp(2.5*(log.(part_radius_stdev)).^2) 
    g = 1 .+ 0.25 .* log.(part_radius_stdev) 

    coeff_of_curve = coeff_of_curve(act_time)
    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    zeta = ((2.*coeff_of_curve)./(3)).*((alpha.*updft_velo)/(diff)).^
           (.5) 

    eta = (  ((alpha.*updft_velo)./(diff)).^(3/2)  ./    
          (2*pi.*WTR_ρ .*gamma.*aero_part_ρ)   )    
          

    
    # Final value:
    MS = sum(((1)./(((S_m).^2) * (f.*((zeta./eta).^(3/2))     
    .+ g.*(((S_m.^2)./(eta+3.*zeta)).^(3/4))))))

    # Comparison between calculations and function output
    @test maxsupersat(part_radius, part_radius_stdev, act_time, 
                      updft_velo, diff, 
                      aero_part_ρ, mass_mx_rat, 
                      diss, 
                      osm_coeff, mass_frac,
                      aero_mm, aero_ρ) 
                      ≈ MS
end
        
@testset "total_N_Act_3" begin
    # Parameters for calculations and function input
    part_radius = [a_SS_ACC, a_SS_COA] 
    act_time = [tau, tau]                                                 
    part_radius_stdev = [sigma_SS_ACC, sigma_SS_COA] 
    updft_velo = [V, V]
    diff = [DIF, DIF] 
    aero_part_ρ = [N, N]
    mass_mx_rat = [r_SS, r_SS]
    diss = [nu_SS, nu_SS]
    osm_coeff = [OC_SS, OC_SS]
    mass_frac = [epsilon_SS, epsilon_SS]
    aero_mm = [M_SS, M_SS]
    aero_ρ = [rho_SS, rho_SS]

    gamma = gamma_sic(M_SS, P_sat) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       


    # Internal calculations
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    coeff_of_curve = coeff_of_curve(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diff, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))

    # Final value:
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Comparison between calculations and function output
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diff, aero_part_ρ) ≈ totN

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
@testset "total_N_Act_5" begin
    # Parameters for calculations and function input
    part_radius = a_SS_ACC 
    act_time = tau                                               
    part_radius_stdev = sigma_SS_ACC
    updft_velo = V 
    diff = DIF  
    aero_part_ρ = 0 # ZERO
    mass_mx_rat = r_SS
    diss = nu_SS
    osm_coeff = OC_SS
    mass_frac = epsilon_SS
    aero_mm = M_SS
    aero_ρ = rho_SS


    gamma = gamma_sic(M_SS, P_sat) # coefficient 
    alpha = alpha_sic(M_SS) # Coefficient in superaturation balance equation       

    # Internal calculations
    B_bar = mean_hygrosopicity(mass_mx_rat, diss, 
                               osm_coeff, mass_frac,
                               aero_mm, 
                               aero_ρ) 

    coeff_of_curve = coeff_of_curve(act_time)

    S_m = S_m(act_time, part_radius, mass_mx_rat, 
              diss, osm_coeff, mass_frac,
              aero_mm, aero_ρ)

    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        updft_velo, diff, 
                        aero_part_ρ, mass_mx_rat, 
                        diss, 
                        osm_coeff, mass_frac,
                        aero_mm, aero_ρ)

    u = ((2*log.(S_m/S_max))./(3.*(2.^.5).*
        log.(part_radius_stdev)))
    
    # Final value:
    totN = sum(aero_part_ρ.*.5.*(1-erf.(u)))

    # Comparison between calculations and function output
    @test total_N_Act(part_radius, act_time, 
                      part_radius_stdev, updft_velo, 
                      diff, aero_part_ρ) ≈ totN

end

#------------------------------------------------------------------------------------------------------
# Test No. 6
# VAL; ONLY total_N_Act_test; DIM 0; Paper validation

# ~~~|This test section checks compares outputs to results give in the 
# ~~~|Abdul-Razzal and Ghan paper to validate the accuracy of the 
# ~~~|functions implemented. 

#------------------------------------------------------------------------------------------------------
# Paper Parameters (AG=Abdul-Razzak+Ghan)
T_AG = 294 # K 
V_AG = 0.5 # m/s
a_AG = 5*10^(-8)
sigma_AG = 2
nu_AG = 3
M_AG = 132 # kg/mol
rho_AG = 1770 # kg/m^3
N_AG = 100000000 # 1/m^3
tau_AG = 1 # s


# Test 6a: Validating initial particle density versus final activation number.

@testset "Validate_initial_N" begin
    # test paramters
    Ns = [810.7472258, 3555.037372, 2936.468114, 
          2515.271387, 1778.226562, 1166.163519]
    totNs = [0.613108201, 0.507426454, 0.525448148, 
             0.542970165, 0.567620911, 0.594486894]
    
    for i in range(size(Ns))
        # Input parameters  TODO: figure out remaining parameters
        part_radius = a_AG # particle mode radius (m)
        act_time = tau_AG # time of activation (s)                                                  
        part_radius_stdev = sigma_AG # standard deviation of mode radius (m)
        updft_velo = V_AG # Updraft velocity (m/s)
        diff = DIF # Diffusion of heat and moisture for particles 
        aero_part_ρ = Ns[i] # Initial particle concentration (1/m^3)
        totN = totNs[i]

        func_totN = total_N_Act(part_radius, act_time, 
                                part_radius_stdev, updft_velo, 
                                diff, aero_part_ρ)

        # Compare results:
        @test  ((totN-functotN)/totN)<.1 
        # ^^^ checks if we are within a certain percent error of paper results

    end


end
