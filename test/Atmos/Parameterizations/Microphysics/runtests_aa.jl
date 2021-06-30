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

WTR_ρ = 1000
WTR_MLR_ρ = 0.022414
WTR_MM = 0.01801528
R = 8.31446261815324
AVO = 6.02214076 × 10^23
G = 9.8
LAT_HEAT_EVP = 2.26 * 10^-6
SPC_HEAT_AIR = 1000
T = 273.1 # K (STP)
P = 100000 # Pa (N/m^2) (STP)

#-----------------------------------------------------------------------------------------------#

@testset "Mean Hygroscopicity" begin
    aero_comp = [1, 2, 3]        
    aero_m_num = [1, 2, 3, 4, 5]
    mass_mx_rat = [[1,    2,    3,    4,    5]
                   [0.1,  0.2,  0.3,  0.4,  0.5]
                   [0.01, 0.02, 0.03, 0.04, 0.05]]
    diss = [[1,   2,   3, 4, 5]
            [0.1, 0.2, 0.3, 0.4, 0.5]
            [0.01, 0.02, 0.03, 0.04, 0.05]]
    osm_coeff = [[1, 2, 3, 4, 5]
           [0.1, 0.2, 0.3, 0.4, 0.5]
           [0.01, 0.02, 0.03, 0.04, 0.05]]
    mass_frac = [[1, 2, 3, 4, 5]
               [0.1, 0.2, 0.3, 0.4, 0.5]
               [0.01, 0.02, 0.03, 0.04, 0.05]]
    aero_mm = [[1, 2, 3, 4, 5]
                        [0.1, 0.2, 0.3, 0.4, 0.5]
                        [0.01, 0.02, 0.03, 0.04, 0.05]]
    aero_ρ = [[1, 2, 3, 4, 5]
                       [0.1, 0.2, 0.3, 0.4, 0.5]
                       [0.01, 0.02, 0.03, 0.04, 0.05]]

    m_h = zeros(3, 5)

    for i in 1:length(aero_comp)
        top_values = mass_mx_rat[i][1] .* diss[i][1] .* osm_coeff[i][1] .* mass_frac[i][1] ./ aero_mm[i]
        top_values *= WTR_MM
        bottom_values = mass_mx_rat[i] ./ aero_ρ[i]
        bottom_values *= WTR_ρ
        m_h[i] = sum(top_values)/ sum(bottom_values)
        top_values = 0
        bottom_values = 0
    end

    @test mean_hygroscopicity(mass_mx_rat[1],
                              diss[1],
                              osm_coeff[1],
                              mass_frac[1],
                              aero_mm[1],
                              aero_ρ[1]
                              ) ≈ m_h[1]
    @test mean_hygroscopicity(mass_mx_rat[2],
                              diss[2],
                              osm_coeff[2],
                              mass_frac[2],
                              aero_mm[2],
                              aero_ρ[2]
                              ) ≈ m_h[2]
    @test mean_hygroscopicity(mass_mx_rat[3],
                              diss[3],
                              osm_coeff[3],
                              mass_frac[3],
                              aero_mm[3],
                              aero_ρ[3]
                              ) ≈ m_h[3]
end


@testset "max_supersat_test" begin

    # constants
    part_radius = [5*(10^(-8))]
    part_radius_stdev = [2]
    act_time = [1]
    alpha = [1]
    updft_velo = [1]
    diff = [1]
    aero_part_ρ = [100000000]
    gamma = [1]

    # Internal calculations:

    # ------ INCOMPLETE-------
    B_bar = mean_hygrosopicity()
    f = 0.5 .* exp(2.5 * (log.(part_radius_stdev)). ^ 2)
    g = 1 .+ 0.25 .* log.(part_radius_stdev)
    A = (2 .* act_time .* WTR_MM)./ (WTR_ρ .* R .* T)
    S_mi = ((2) ./ (B_i_bar) .^ (.5)) .* ((A) ./ (3.*part_radius)) .^ (3/2)
    zeta = ((2 .* A) ./ (3)).*((alpha .* updft_velo)/(diff)) .^ (.5)
    eta = (((alpha .* updft_velo) ./ (diff)) .^ (3/2)
          ./ (2 * pi .* WTR_ρ .* gamma .* aero_part_ρ))

    # Final value for maximum supersaturation:
    MS = sum(((1) ./ (((S_mi) .^ 2) * (f_i .* ((zeta ./ eta) .^ (3/2))
         + g_i .* (((S_mi .^ 2)./(eta + 3 .* zeta)) .^ (3/4))))))

    # Comaparing calculated MS value to function output: 
    @test maxsupersat(part_radius, part_radius_stdev, act_time, 
                      WTR_MM, WTR_ρ, R, T, 
                      B_i_bar, alpha, updft_velo, diff, 
                      aero_part_ρ, gamma) ≈ MS
end

@testset "smallactpartdryrad_test" begin
    # constants

    part_radius = [5*(10^(-8))]
    act_time = [1]
    part_radius_stdev = [2]
    alpha = [1]
    updft_velo = [1]
    diff = [1]
    aero_part_ρ = [100000000]
    gamma = [1]
    
    # Internal calculations:

    # ------ INCOMPLETE-------
    B_bar = mean_hygrosopicity()
    A = (2 .* act_time .* water_molecular_mass) ./ (WTR_ρ .* R .* T)
    S_mi = ((2) ./ (B_i_bar) .^ (.5)) .* ((A) ./ (3.*part_radius)) .^ (3/2)
    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        WTR_MM, WTR_ρ , R, T, 
                        B_i_bar, alpha, updft_velo, diff, 
                        aero_part_ρ, gamma)
    
    # Final value for dry radius:
    drsap = particle_radius.*((S_mi)./(S_max)).^(2/3)
    
    # Comaparing calculated dry radius value to function output: 
    @test smallactpartdryrad(part_radius, part_radius_stdev, 
                             act_time, R, T, B_i_bar, 
                             alpha, updft_velo, diff, 
                             aero_part_ρ, gamma) ≈ drsap
    
    end
        
    

@testset "total_N_Act_test" begin
    # constants
    part_radius = [5*(10^(-8))]
    act_time = [1]
    R = [8.31446261815324]
    part_radius_stdev = [2]
    alpha = [1]
    updft_velo = [1]
    diff = [1]
    aero_part_ρ = [100000000]
    gamma = [1]
    
    # Internal calculations:

    # ------ INCOMPLETE-------
    B_bar = mean_hygrosopicity() 
    A = (2 .* act_time .* WTR_MM) ./ (WTR_ρ .* R .* T) 
    S_min = ((2) ./ (B_bar) .^ (.5)) .* ((A) ./ (3 .* part_radius)) .^ (3/2) 
    S_max = maxsupersat(part_radius, part_radius_stdev, act_time, 
                        WTR_MM, WTR_ρ , R, T, 
                        B_i_bar, alpha, updft_velo, diff, 
                        aerosol_particle_density, gamma)
    u = ((2 * log.(S_min/S_max)) ./ (3.*(2 .^ .5) .* log.(part_radius_stdev)))

    # Final value for total number:
    totN = sum(aero_part_ρ .* .5 .* (1-erf .(u)))

    # Comaparing calculated total number value to function output:
    @test total_N_Act(part_radius, act_time, WTR_MM, 
                      WTR_ρ , R, T, B_i_bar, 
                      part_radius_stdev, alpha, updft_velo, diff, 
                      aero_part_ρ, gamma) ≈ totN

end

@testset alpha()
    # constants
    aero_mm = 0.058443

    # Final value for alpha:
    a = (G * WTR_MM * LAT_HEAT_EVP) 
        / (SPC_HEAT_AIR * R * T^2) - G * aero_mm / (R * T)
    
    # Comaparing calculated alpha value to function output:
    @test alpha_sic(T, aero_mm) ≈ a
end

@testset gamma()
    # constants
    aero_mm = 0.058443
    aero_ρ = 2170

    # Final value for gamma:
    g = R * T / (aero_ρ * WTR_MM) + WTR_MM 
        * SPC_HEAT_AIR ^ 2 / (SPC_HEAT_AIR * WTR_ρ * aero_mm * T)
    
    # Comaparing calculated alpha value to function output:
    @test gamma_sic(T, aero_mm, aero_mm) ≈ g
end

end 