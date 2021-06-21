"""
    Activation

TODO
"""
module Activation

using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Atmos.Microphysics_0M
const APS = AbstractParameterSet

export activation_cloud_droplet_number

"""
    activation_cloud_droplet_number(param_set::APS, q, T, ρ)

TODO
"""
function activation_cloud_droplet_number(
    param_set::APS,
    q::PhasePartition{FT},
    T::FT,
    ρ::FT,
) where {FT <: Real}

    _τ_precip::FT = τ_precip(param_set)
    _qc_0::FT = qc_0(param_set)

    # TODO
    water_density::FT = 42

    return 42
end

end #module Activation.jl
