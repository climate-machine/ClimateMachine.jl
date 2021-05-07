using ClimateMachine
using Test
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.Mesh.Interpolation
using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
import CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 1
function initialize_planet_size(domain_height)
    FT, n_horz, n_vert, poly_order = Float64, 12, 6, 3
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = x -> x,
    )
    _planet_radius = FT(planet_radius(param_set))
    driver_config = ClimateMachine.AtmosGCMConfiguration(
        "SmallPlanet",
        poly_order,
        (n_horz, n_vert),
        domain_height,
        param_set,
        x -> x;
        model = model,
    )
    info = driver_config.config_info
    boundaries = [
        FT(-90) FT(-180) _planet_radius
        FT(90) FT(180) FT(_planet_radius + info.domain_height)
    ]
    resolution = (FT(2), FT(2), FT(1000)) # in (deg, deg, m)
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    return nothing
end

let
    ClimateMachine.init()
    for planet_rad in 10.0 .^ (0:7)
        CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = planet_rad
        for aspect_ratio in 10.0 .^ (-2.33:0.6:0.3)
            domain_height = aspect_ratio * planet_radius(param_set)
            initialize_planet_size(domain_height)
        end
    end
end

nothing
