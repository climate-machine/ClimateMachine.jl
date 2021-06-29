TD.PhaseEquil_dev_only(
    param_set,
    e_int[5],
    ρ[5],
    q_tot[5];
    sat_adjust_method = RegulaFalsiMethod,
    maxiter = 10,
)

e_int_prof = profiles.e_int;
ρ_prof = profiles.ρ;
q_tot_prof = profiles.q_tot;
Thermodynamics.print_warning() = true

TD.PhaseEquil_dev_only.(
    param_set,
    e_int_prof,
    ρ_prof,
    q_tot_prof;
    sat_adjust_method = RegulaFalsiMethod,
    maxiter = 10,
)
