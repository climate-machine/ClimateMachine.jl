# Thermodynamics

## Saturation adjustment

The saturation adjustment procedure requires solving a non-linear
equation.

In [Thermodynamics-docs](@ref), we plotted the thermodynamic-tested
profiles. In this section, we review our success/failure rate of
thermodynamic states outside of this manifold. In particular, rather
than being interested in physically meaningful combinations of
the inputs, we are interested in all permutations of inputs within
a given range of `ρ, T, q_tot`. Some of these permutations may
not be physically meaningful, or likely to be observed in climate
simulations, but showing the convergence space helps illustrate the
buffer between our tested profiles and the nearest space where
convergence fails.

This section is dedicated to monitoring the status and improvement
of the performance and robustness of various numerical methods
in solving the saturation adjustment equations for various thermodynamic
formulations.

```@example
include("saturation_adjustment.jl")
```

## 3D space
| Numerical method  | Converged  |  Non-converged |
:-----------------:|:-----------------:|:---------------------:
SecantMethod | ![](3DSpace_converged_SecantMethod.svg)       |  ![](3DSpace_non_converged_SecantMethod.svg)
NewtonsMethod | ![](3DSpace_converged_NewtonsMethod.svg)      |  ![](3DSpace_non_converged_NewtonsMethod.svg)
NewtonsMethodAD | ![](3DSpace_converged_NewtonsMethodAD.svg)    |  ![](3DSpace_non_converged_NewtonsMethodAD.svg)
RegulaFalsiMethod | ![](3DSpace_converged_RegulaFalsiMethod.svg)  |  ![](3DSpace_non_converged_RegulaFalsiMethod.svg)

## 2D slices, binned by total specific humidity

| Numerical method  | Converged  |  Non-converged |
:-----------------:|:-----------------:|:---------------------:
SecantMethod | ![](2DSlice_converged_SecantMethod.svg)  |  ![](2DSlice_non_converged_SecantMethod.svg)
NewtonsMethod | ![](2DSlice_converged_NewtonsMethod.svg)  |  ![](2DSlice_non_converged_NewtonsMethod.svg)
NewtonsMethodAD | ![](2DSlice_converged_NewtonsMethodAD.svg)  |  ![](2DSlice_non_converged_NewtonsMethodAD.svg)
RegulaFalsiMethod | ![](2DSlice_converged_RegulaFalsiMethod.svg)  |  ![](2DSlice_non_converged_RegulaFalsiMethod.svg)

