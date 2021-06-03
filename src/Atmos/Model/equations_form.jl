export Unsplit, CentralSplitForm, KennedyGruberSplitForm

abstract type AbstractEquationsForm end
struct Unsplit <: AbstractEquationsForm end

abstract type AbstractKennedyGruberSplitForm <: AbstractEquationsForm end
struct KennedyGruberSplitForm <: AbstractKennedyGruberSplitForm end
struct KennedyGruberGravitySplitForm <: AbstractKennedyGruberSplitForm end

two_point_flux(
    var::AbstractPrognosticVariable,
    td::TendencyDef{Flux{FirstOrder}},
    atmos::AtmosModel,
    args,
) = two_point_flux(atmos.equations_form, var, td, atmos, args)

# used for testing, central flux differencing should recover standard DGSEM
struct CentralSplitForm <: AbstractEquationsForm end
function two_point_flux(
    ::CentralSplitForm,
    var::AbstractPrognosticVariable,
    td::TendencyDef{Flux{FirstOrder}},
    atmos::AtmosModel,
    args,
)
    @unpack state1, aux1, state2, aux2 = args

    _args1 = (state = state1, aux = aux1)
    args1 = merge(
        _args1,
        (precomputed = precompute(atmos, _args1, Flux{FirstOrder}()),),
    )
    flux1 = flux(var, td, atmos, args1)

    _args2 = (state = state2, aux = aux2)
    args2 = merge(
        _args2,
        (precomputed = precompute(atmos, _args2, Flux{FirstOrder}()),),
    )
    flux2 = flux(var, td, atmos, args2)

    return (flux1 + flux2) / 2
end
