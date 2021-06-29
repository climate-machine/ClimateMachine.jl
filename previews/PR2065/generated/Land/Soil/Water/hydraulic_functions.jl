using Plots

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations

const FT = Float32;

vg_α = FT(7.5) # m^-1
vg_n = FT(1.89)
hydraulics = vanGenuchten{FT}(α = vg_α, n = vg_n)

ψ_sat = 0.09 # m
Mval = 0.228
hydraulics_bc = BrooksCorey{FT}(ψb = ψ_sat, m = Mval);

S_l = FT.(0.01:0.01:0.99)
ψ = matric_potential.(Ref(hydraulics), S_l)
ψ_bc = matric_potential.(Ref(hydraulics_bc), S_l)
plot(
    S_l,
    log10.(-ψ),
    xlabel = "effective saturation",
    ylabel = "Log10(|ψ|)",
    label = "van Genuchten",
)
plot!(S_l, log10.(-ψ_bc), label = "Brooks and Corey")
savefig("bc_vg_matric_potential.png")

ν = FT(0.41)
Ksat = FT(4.42 / (3600 * 100))
moisture_choice = MoistureDependent{FT}()
viscosity_choice = ConstantViscosity{FT}()
impedance_choice = NoImpedance{FT}();

T = FT(0.0)
θ_i = FT(0.0)

K =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice),
        Ref(moisture_choice),
        Ref(hydraulics),
        Ref(θ_i),
        Ref(ν),
        Ref(T),
        S_l,
    );

viscosity_choice_T = TemperatureDependentViscosity{FT}()
T = FT(300.0)
K_T =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice_T),
        Ref(moisture_choice),
        Ref(hydraulics),
        Ref(θ_i),
        Ref(ν),
        Ref(T),
        S_l,
    )
ice_impedance_I = IceImpedance{FT}()
θ_i = FT(0.1)
S_i = θ_i / ν;

S_l_accounting_for_ice = FT.(0.01:0.01:(0.99 - S_i))
K_i =
    Ksat .*
    hydraulic_conductivity.(
        Ref(ice_impedance_I),
        Ref(viscosity_choice),
        Ref(moisture_choice),
        Ref(hydraulics),
        Ref(θ_i),
        Ref(ν),
        Ref(T),
        S_l_accounting_for_ice,
    )
plot(
    S_l,
    log10.(K),
    xlabel = "total effective saturation, (θ_i+θ_l)/ν",
    ylabel = "Log10(K)",
    label = "Base case",
    legend = :bottomright,
)
plot!(S_l, log10.(K_T), label = "Temperature Dependent Viscosity; no ice")
plot!(
    S_l_accounting_for_ice .+ S_i,
    log10.(K_i),
    label = "Ice Impedance; S_i=0.24",
)
savefig("T_ice_K.png")

T = FT(0.0)
θ_i = FT(0.0)

K_bc =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice),
        Ref(moisture_choice),
        Ref(hydraulics_bc),
        Ref(θ_i),
        Ref(ν),
        Ref(T),
        S_l,
    )
plot(
    S_l,
    log10.(K),
    xlabel = "effective saturation",
    ylabel = "Log10(K)",
    label = "van Genuchten",
)
plot!(
    S_l,
    log10.(K_bc),
    xlabel = "effective saturation",
    ylabel = "Log10(K)",
    label = "Brooks and Corey",
)
savefig("bc_vg_k.png")

no_moisture_dependence = MoistureIndependent{FT}()
K_constant =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice),
        Ref(no_moisture_dependence),
        Ref(hydraulics),
        Ref(θ_i),
        Ref(ν),
        Ref(T),
        S_l,
    );

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

