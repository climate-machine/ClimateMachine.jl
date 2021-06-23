@testset "smallactpartdryrad_test" begin
# Parameter inputs:
a_m = [5*(10^(-8))] # particle mode radius (m)
tau = [1] # time of activation (s)                                                  
M_w = [0.01801528] # Molecular weight of water (kg/mol)
rho_w = [1000] # Density of water (kg/m^3)
R = [8.31446261815324] # Gas constant (kg)
T = [273.15] # Temperature (K)
sigma = [2] # standard deviation of mode radius (m)
alpha = [1] # Coefficient in superaturation balance equation       
V = [1] # Updraft velocity (m/s)
G = [1] # Diffusion of heat and moisture for particles 
N = [100000000] # Initial particle concentration (1/m^3)
gamma = [1] # coefficient 

# Internal calculations:
B_bar = mean_hygrosopicity() # calculated in earlier function    ------ INCOMPLETE-------
A = (2.*tau.*M_w)./(rho_w.*R.*T) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
S_min = ((2)./(B_i_bar).^(.5)).*((A)./(3.*a_m)).^(3/2) # Minimum supersaturation
S_max = maxsupersat(a_m, sigma tau, M_w, rho_w, R, T, B_i_bar, alpha, V, G, N, gamma)

# Final calculation:
DRSAP = a_m.*((S_mi)./(S_max)).^(2/3)

# Running test:
@test smallactpartdryrad(a_m, sigma, tau, M_w, rho_w, R, T, B_i_bar, alpha, V, G, N, gamma) = DRSAP

end
    
