import ClimateMachine.Mesh.Grids: _x3

struct CplTestModel{G, D, B, S, TS}
    grid::G
    discretization::D
    boundary::B
    state::S
    stepper::TS
    nsteps::Int
end

"""
    CplTestModel(;domain,
                  equations,
                  nsteps,
                  btags=((0,0),(0,0),(1,2) ),
                  dt=1.,
                  NFSecondOrder=CentralNumericalFluxSecondOrder())

Builds an instance of a coupler test model.  This is a toy model
used for testing and designing coupling machinery. In a full-blown coupled
experiment this model would be replaced by a full compnent model.

      -  `domain` the spatial mesh used by this model. 
      -  `equations` the Balance Law used by this model.
      -  `nsteps` number of component steps to run during each coupling step.
      -  `btags` boundary labels for different edges.
      -  `dt` component timestep to use on each component step.
      -  `NFSecondOrder` numerical flux to use for second order terms.

Each returned model instance is independent and has its own grid,
balance law, time stepper and other attributes.  For now the code
keeps some of these things the same for initial testing, including
component timestepper and initial time (both of which need tweaking
to use for real setups).

The argumetns and btags and NFSecondOrder are useful for this particular model. A
real model might have many more flags and/or may wrap the component creation
very differently. Any component should allow itself to set a number of timesteps
to execute with a certain timestep to synchronize with the coupling time scale.

"""
function CplTestModel(;
    domain,
    equations,
    nsteps,
    btags = ((0, 0), (0, 0), (1, 2)),
    dt = 1.0,
    NFSecondOrder = CentralNumericalFluxSecondOrder(),
)

    FT = eltype(domain)

    ###
    ### Instantiate the spatial grid for this component.
    ### Use ocean scripting interface convenience wrapper for now.
    ###
    grid = nothing
    grid = DiscontinuousSpectralElementGrid(domain; boundary_tags = btags)

    ###
    ### Create a discretization that is the union of the spatial
    ### grid and the equations, plus some numerical flux settings.
    ###
    discretization = DGModel(
        equations,
        grid,
        RusanovNumericalFlux(),
        NFSecondOrder,
        # CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
        # direction = VerticalDirection(),
    )

    # Specify the coupling boundary
    boundary = grid.vgeo[:, _x3:_x3, :] .== 0

    ###
    ### Invoke the spatial ODE initialization functions
    ###
    state = init_ode_state(discretization, FT(0); init_on_cpu = true)

    ###
    ### Create a timestepper of the sort needed for this component.
    ### Hard coded here - but can be configurable.
    ###
    stepper = LSRK54CarpenterKennedy(discretization, state, dt = dt, t0 = 0.0)

    ###
    ### Return a CplTestModel entity that holds all the information
    ### for a component that can be driver from a coupled stepping layer.
    ###
    return CplTestModel(grid, discretization, boundary, state, stepper, nsteps)
end
