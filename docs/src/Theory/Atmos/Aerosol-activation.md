# Aerosol Activation Parametrization

The `Activation.jl` module defines a parameterization of
  the activation of multiple lognormal modes with a uniform internal mixture 
  composed of both soluble and insoluble materials.
It uses the combination of previous studies and the Köhler theory to model the formation of cloud droplets.

Aerosol activation is when cloud particles form on the aerosol particles existing in the air.

Assumptions:
  - There are no kinetic limitations on the aerosol activation process.
  - The particles are activated if their critical supersaturation is less than the maximum supersaturation in the numerical integration.
  - Do not consider the effects of surfactants on surface tension.
  - The solute is sufficiently soluble so its concentration does not increase as the droplet grows.

Other notes about the parametrization:
  - Using a lognormal aerosol size distribution to derive the parametrization that is naturally bounded at high updraft velocity.
  - Worked with 2 broad aerosol modes rather than many more narrow ones. Each mode has an interal mixture of material and they all compete for water.

The following are equations used for the parametrization.

1: Hydgroscopicity Parameter
```math
B \equiv \frac{v \phi \epsilon M_{w} \rho_a}{M_a \rho_{w}}
```
where
  - ``B`` is the hygroscopicity parameter.
  - ``v`` is the number of ions salt disassociates into when in water.
  - ``\phi`` is the osmotic coefficient.
  - ``\epsilon`` is the mass fraction of soluble material.
  - ``M_{w}`` is the molecular weight of water.
  - ``\rho_{a}`` is the density of the aerosol material.
  - ``M_{a}`` is the molecular weight of the aerosol material.
  - ``p_{w}`` is the density of water.

2: Mean Hygroscopicity Paramter
```math
\bar{B_i} \equiv \frac{M_{w} \sum_{j = 1}^{J} r_{ij} v_{ij} \phi_{ij} \epsilon_{ij} /M_{aij}} {\rho_{w} \sum_{j = 1}^{J} r_{ij}/p_{aij}}
```
where
  - ``\bar{B_i}`` is the mean hygroscopicity parameter of the aerosol material in the aerosol mode ``i``.
  - ``v_{ij}`` is the number of ions that salt disassociates into when in water of aerosol component ``j`` and mode ``i``.
  - ``\phi_{ij}`` is the osmotic coefficient of aerosol component ``j`` and mode ``i``.
  - ``\epsilon_{ij}`` is the mass fraction of the soluble material of aerosol component ``j`` and mode ``i``.
  - ``M_w`` is the molecular weight of water.
  - ``\rho_{aij}`` is the density of the aerosol's component ``j`` and mode ``i``.
  - ``M_{aij}`` is the molecular weight of aerosol component ``j`` and mode ``i``.
  - ``\rho_{w}`` is the density of water.
  - ``j`` is the component number. Min value is 1 and max value is ``J``.
  - ``r_{ij}`` is the mass mixing ratio of component ``j`` and mode ``i``.
  - ``i`` is the mode number.

3: Condensation Rate
```math
\frac{dw}{dt} = 4 \pi \rho_{w} \sum_{i = 1}^{I} \int_{0}^{S} r_i^2 \frac{dr_i}{dt} \frac{dn_i(S')}{da} dS'
```
where
  - ``i`` is the aerosol type. The min is 1 and the max is ``I``.
  - ``r_{i}`` is the radius of the droplet forming on aerosol type ``i``.
  - ``S`` and ``S'`` represent the supersaturation.
  - ``n_i`` is the number concentration of aerosol mode i.
  - ``w`` is the condensation.
  - ``\rho_w`` is the density of water.
  - ``a`` is the geometric mean dry radius.

4: Maximum/Critical Supersaturation
```math
S_{max} = \frac{1}{{\sum_{i=1}^{I} \frac{1}{S_{mi}^{2}} [f_i(\frac{\zeta}{\eta_{i}})^{\frac{3}{2}} + g_{i}(\frac{S_{mi}^{2}}{\eta_{i} + 3\zeta})^{\frac{3}{4}}}]^{\frac{1}{2}}}

```
where
  - ``S_{mi} = \frac{2}{\sqrt{\bar{B_{i}}}}(\frac{A}{3a_{mi}})^{3/2}``
  - ``\zeta \equiv \frac{2A}{3}(\frac{\alpha V}{G-})^{\frac{1}{2}}``
  - ``f_i \equiv 0.5 \mathrm{exp} (2.5 \mathrm{ln}^{2} \sigma_{i})``
  - ``\eta_{i} \equiv \frac{(\alpha V/G)^{\frac{3}{2}}}{2 \pi \rho_{w} \gamma N_{i}}``
  - ``g_i \equiv 1 + 0.25 \mathrm{ln} \sigma_i``
  - ``A`` is the surface tension effects in the Köhler equilibrium equation.
  - ``V`` is the updraft velocity.
  - ``G`` is the diffusion of heat and moisture to the particles.
  - ``\alpha`` and ``\gamma`` are coefficients in the supersaturation balance equation.

5: Dry Radius
```math
a_{ci} = a_{mi}(\frac{S_{mi}}{S_{max}})^{\frac{2}{3}}       
```
where
  - ``i = 1,...,I``
  - ``a_{mi}`` is the geometric mean dry radius.
  - ``S_{mi} = \frac{2}{\sqrt{\bar{B_{i}}}}(\frac{A}{3a_{mi}})^{3/2}``
  - ``S_{max}`` is the equation 4.

6: Total Number of Activated Aerosols
```math
N = \sum_{i = 1}^{I} N_{i}\frac{1}{2}[1 - \mathrm{erf}(u_{i})]
```
where
  - ``N_{i}`` is the total number concentration.
  - ``u_{i} \equiv \frac{\mathrm{ln}(a_{ci}/a_{mi})}{\sqrt{2} \mathrm{ln} \sigma_{i}} = \frac{2 \mathrm{ln}(S_{mi}/S_{max})}{3\sqrt{2} \mathrm{ln} \sigma_{i}}``
  - ``\sigma_{i}`` is the geometric standard deviation for aerosol mode ``i``.

7: Total Mass of Activated Aerosols
```math
M = \sum_{i = 1}^{I} M_{i}\frac{1}{2}[1 - \mathrm{erf}(u_{i} - \frac{3\sqrt{2}}{2} \mathrm{ln}\sigma_{i})]
```
where
  - ``u_{i} \equiv \frac{\mathrm{ln}(a_{ci}/a_{mi})}{\sqrt{2} ln \sigma_{i}} = \frac{2 \mathrm{ln}(S_{mi}/S_{max})}{3\sqrt{2} ln \sigma_{i}}``
  - ``\sigma_{i}`` is the geometric standard deviation for aerosol mode ``i``.
  - ``M_{i}`` is the mass of the aerosol mode ``i``.

8: S_mi Calculation
```math
S_mi = \frac{2}{\bar{B_i}}^{0.5} * \frac{3}{part_rds}^{3/2}
```
where
  - ``S_mi`` is the supersaturation of the aerosol mode ``i``.
  - ``\bar{B_i}`` is the mean hygroscopicity parameter of the aerosol material in the aerosol mode ``i``.
  - ``part_rds`` is the mean radius of the particles.

9: A Calculation
```math
A = \frac{2 * act_time .* wtr_mm}{wtr_ρ .* R .* temp}
```
where
  - ``act_time`` is time that the aerosol gets activated. 
  - ``water_molecular_mass`` is the molecular mass of water. 
  - ``water_density`` is the density of water. 
  - ``R`` is the universal gas constant. 
  - ``temperature`` is the temperature.

10: Alpha Calculations (Size Invariant Coefficients)
```math
\alpha = \frac{gravity * water_molecular_mass * latent_heat_evaporation}
         {specific_heat_air * R * T^2} - \frac{gravity * M_a}{R * T}
```
where 
  - ``gravity`` is the gravitational force of the Earth.
  - ``water_molecular_mass`` is the molecular mass of water. 
  - ``latent_heat_evaporation`` is the latent heat of evaporation.
  - ``specific_heat_air`` is the specific heat of air.
  - ``R``  is the universal gas constant. 
  - ``T`` is the temperature
  - ``M_{a}`` is the molecular weight of the aerosol material.

11: Gamma Calculations (Size Invariant Coefficients)
```math
\gamma = R * T / (rho_{s} * water_molecular_mass) + water_molecular_mass 
        * specific_heat_air ^ 2 / (specific_heat_air * water_density * M_a * T)
```
  - ``water_molecular_mass`` is the molecular mass of water. 
  - ``specific_heat_air`` is the specific heat of air.
  - ``water_density`` is the density of water.
  - ``R``  is the universal gas constant. 
  - ``T`` is the temperature
  - ``M_{a}`` is the molecular weight of the aerosol material.
  - ``rho_{a}`` is the desnity of the aerosol.

Variables:
  - ``osmotic_coefficient`` 
  - ``temperature``
  - ``aerosol_molecular_mass``
  - ``aerosol_particle_density``
  - ``aerosol_density``
  - ``water_density``
  - ``water_molar_density``
  - ``water_molecular_mass``
  - ``updraft_velocity``
  - ``R``
  - ``avogadros_number``
  - ``dissociation``
  - ``mass_fraction``
  - ``particle_radius``
  - ``particle_radius_stdev``
  - ``activation_time``
  - ``alpha`` # ADD MORE
  - ``diffusion``
  -  ``gamma`` # ADD MORE

| Variable Name - Code | Meaning                               | Units
Variable Name - Paper
|  - TEMP              | temperature                           | Kelvin
|  - WTR_MM            | molecular mass of water               | kg/mol
|  - WTR_MLR_ρ         | molar density of water                | m^3/mol
|  - WTR_ρ             | water density                         | kg/m^3
|  - R                 | gas constant                          | (kg m^2) / (s^2 K mol)
|  - AVO               | water density                         | kg/m^3
|  - G                 | gravity                               | m/s^2
|  - LAT_HEAT_EVP      | latent heat of evaporation            | J/kg
|  - SPC_HEAT_AIR      | specific heat of air                  | J/kg
|  - P                 | pressure                              | Pa
|  - act_time          | activation time                       | seconds
|  - part_radius       | particle radius                       | meters
|  - mass_mx_rat       | mass mixing ratio                     | N/A
|  - diss              | dissociation                          | N/A
|  - osm_coeff         | osmotic coefficient                   | N/A
|  - mass_frac         | mass fraction                         | N/A
|  - aero_mm           | molecular mass of aerosol             | kg/mol
|  - aero_ρ            | density of aerosol                    | kg/m^3
|  - a                 | alpha                                 | ?
|  - b_bar             | mean hygroscopicity                   | ? 
|  - s_m               | supersaturation FILL                  | ?
|  - part_radius_stdev | standard deviation of particle radius | meters
|  - updft_velo        | updraft velocity                      | m/s
|  - diff              | diffusion                             | ?
|  - aero_part_ρ       | aerosol particle density              | kg/m^3
|  - f                 | ?                                     | ?
|  - g                 | ?                                     | ?
|  - gamma             | size invariant coefficients           | N/A
|  - zeta              | dimentionless constant                | N/A
|  - eta               | dimentionless constant                | N/A
|  - s_max             | maximum supersaturation               | ?
|  - u                 | ?                                     | ?
|  - totN              | total number                          | N/A
|  - P_sat             | saturation pressure                   | Pa
|  - K 
|  - R_v
|  - D
