using SpecialFunctions

function maxsupersat(a_mi, sigma_i, tau, M_w, rho_w, R, T, B_i_bar, alpha, V, G, N_i, gamma)

    # Internal calculations: 
    f_i = 0.5*exp(2.5*(log(sigma_i))^2)
    g_i = 1 + 0.25 * log(sigma_i)        

    A = (2*tau*M_w)/(rho_w*R*T)

    S_mi = (2/(B_i_bar)^.5)*(A/(3*a_mi))^(3/2)
    
    zeta = ((2*A)/3)*((alpha*V)/G)^(.5)
    eta = (((alpha*V)/G)^(3/2))/(2*pi*rho_w*gamma*N_i)

    # Final calculation:
    mss = 1/(((1/S_mi^2)*((f_i*(zeta/eta)^(3/2))+(g_i*((S_mi^2)/(eta+3*zeta))))^.5)


    return mss


end


function total_N_Act(a_mi, tau, M_w, rho_w, R, T, B_i_bar, sigma_i, alpha, V, G, N_i, gamma, S_max)

    # Internal calculations:
    A = (2*tau*M_w)/(rho_w*R*T)

    S_mi = (2/(B_i_bar)^.5)*(A/(3*a_mi))^(3/2)
    
    u = (2*log(S_mi/S_max))/(3*(2^.5)*log(sigma_i))
    
    # Final Calculation: 
    
    N = N_i*.5*(1-erf(u))

    return N

end
