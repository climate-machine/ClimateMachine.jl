using Unitful, Dates

# TODO: Build constructor that uses a model component's grid.
struct CplGridInfo{GT, GP, GH, GE}
    gridtype::GT
    gridparams::GP
    gridhandle::GH
    gridencoding::GE
end

mutable struct CplFieldInfo{AT, GT <: CplGridInfo, UT <: Unitful.Units}
    data::AT
    gridinfo::GT
    units::UT
    datetime::DateTime
end

mutable struct CplState{DT}
    CplStateBlob::DT
end

"""
    CplState()

Type for holding Coupler "state". This is the namespace through which coupled components
communicate. Its role is to provide a level of indirection so that components remain modular
and so that any data communication, interpolation, reindexing/unit conversions and filtering 
etc... can be embeded in the intermdediate coupling layer.

To start with we can just use a dictionary key and value table that holds labelled pointers to various fields.
A field is exported by one component and imported by one or more other components. Components
can select which fields are needed by using the Dict symbols.
"""
function CplState()
    return CplState(Dict{Symbol, CplFieldInfo}())
end

"""
    register_cpl_field!(
            coupler::CplState,
            fieldname::Symbol,
            fieldvalue,
            grid,
            datetime::DateTime,
            units::Unitful.Units = Unitful.NoUnits, 
        )

Add a field to the coupler that is accessible with key `fieldname`. 

# Arguments
- `coupler`: coupler object the field is registered to.
- `fieldname`: key to access the field in the coupler.
- `fieldvalue`: data array of field values.
- `grid`: grid the field is stored on.
- `datetime`: time associated with the field state.
- `units`: units associated with the field values. Dimensionless by default.
"""
function register_cpl_field!(
        coupler::CplState,
        fieldname::Symbol,
        fieldvalue,
        grid,
        datetime::DateTime,
        units::Unitful.Units = Unitful.NoUnits, 
    )
    gridinfo = CplGridInfo(nothing, nothing, nothing, nothing)
    push!(coupler.CplStateBlob, fieldname => CplFieldInfo(fieldvalue, gridinfo, units, datetime))
end

"""
    get(coupler::CplState, fieldname::Symbol, gridinfo, datetime::DateTime, units::Unitful.Units)

Retrieve data array corresponding to `fieldname`.

Returns data on the grid specified by `gridinfo` and in the units of `units`. Checks that
the coupler data field is the state at time `datetime`.
"""
function get(coupler::CplState, fieldname::Symbol, gridinfo, datetime::DateTime, units::Unitful.Units)
    cplfield = coupler.CplStateBlob[fieldname]

    # check that retrieving component and coupler are at same time
    @assert datetime == cplfield.datetime "Retrieval time ($datetime) != coupler field time ($(cplfield.datetime))"

    regriddata = regrid(cplfield.data, gridinfo, cplfield.gridinfo)

    return uconvert(regriddata, units, cplfield.units)
end

"""
    put!(coupler::CplState, fieldname::Symbol, fieldvalue, gridinfo, datetime::DateTime, units::Unitful.Units)

Updates coupler field `fieldname` with `fieldvalue`, the field's value at time `datetime`.

`gridinfo` and `units` inform the coupler of the format of the inputted data
allowing conversion to match the grid and units of the coupler field.
"""
function put!(coupler::CplState, fieldname::Symbol, fieldvalue, gridinfo, datetime::DateTime, units::Unitful.Units)
    cplfield = coupler.CplStateBlob[fieldname]

    # map new data to grid of coupler field; new data -> coupler grid
    regriddata = regrid(fieldvalue, cplfield.gridinfo, gridinfo)

    # store new regridded data in correct units
    cplfield.data .= uconvert(regriddata, cplfield.units, units)

    # update timestamp of coupler field
    settime!(cplfield, datetime)

    return nothing
end

# TODO: maps cplfield data from `fromgrid` to `togrid`
function regrid(data, togrid, fromgrid)
    return data
end

# convert from `fromunits` to `tounits`
# ustrip throws error if the units aren't dimensionally compatible
function uconvert(data, tounits::Unitful.Units, fromunits::Unitful.Units)
    tounits == fromunits && return data
    return ustrip.(tounits, data .* fromunits)
end

# set the time of the coupled field
function settime!(cplfield::CplFieldInfo, newtime::DateTime)
    cplfield.datetime = newtime
end