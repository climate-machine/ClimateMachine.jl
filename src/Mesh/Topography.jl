module Topography
#
#    # Topography:
#
#    This module contains:
#
#    -  ReadExternalHeader(header_file_in)
#    -  ReadExternalTxtCoordinates(body_file_in, TopoBathy_flg, nlon, nlat)
#
#    to read extenral grid files. 
#
#    ## The current implementation can only read ASCII files from the NOAA database
#        url{https://www.ngdc.noaa.gov/mgg/topo/}
#
#
import MPI

using Printf
using DelimitedFiles
using Dierckx

export ReadExternalHeader, ReadExternalTxtCoordinates

# {{{ READTOPOtxt_header
"""   
    ReadExternalHeader(header_file_in)
  
    reads the header file containing the details of the topography from 
    - NOAA ASCII file
    - (NCDF and DEM files to be added. See reader in MMesh3D)
  
    
    returns nlon, nlat, lonmin, lonmax, latmin, latmax, dlon, dlat

 
"""
function ReadExternalHeader(header_file_in)

    @info @sprintf """ Topography header file %s ... DONE""" header_file_in

    open(header_file_in, "r") do ftopo_header
        @info @sprintf """ Grids.jl: Opening topography header file ... DONE"""
        topo_header = readdlm(ftopo_header)
        @info @sprintf """ Grids.jl: Reading topography header file ... DONE"""
        #Check that the file is not empty
        (nrows, _) = size(topo_header)
    end
    @info @sprintf """ Grids.jl: Closing topography header file ... DONE"""


    if nrows == 0
        error(" ERROR Grids.jl --> READTOPOtxt_header: grid header file is empty!")
    end

    nlon = IT(topo_header[1, 2])
    nlat = IT(topo_header[2, 2])
    lonmin = FT(topo_header[3, 2])
    latmin = FT(topo_header[4, 2])
    deltacell = FT(topo_header[5, 2])
    lonmax = lonmin + nlon * deltacell
    latmax = latmin + nlat * deltacell

    dlon = lonmax - lonmin
    dlat = latmax - latmin

    return (nlon, nlat, lonmin, lonmax, latmin, latmax, dlon, dlat)

end
# }}}


# {{{ READTOPOtxt_header
"""
    ReadExternalTxtCoordinates(body_file_in, TopoBathy_flg, nlon, nlat)

Reads the topography from a NOAA text file of shape [1:nnodes][3]
where the first and second column are the ordered lat-lon coordinates
and the third column is the height of topography at that specific
coordinate point.
                     
1) XYZ files from NOAA
                       
READTOPOtxt_header() reads the parameters from the header file (*.hdr)
READTOPOtxt_file()   reads the actual file of coordinates      (*.xyz)
                         
2) DEM files

READTOPO_DEM()       reads a DEM file from NOAA page (file extension: *.asc)

returns xTopo1d, yTopo1d, zTopo2d
"""
function ReadExternalTxtCoordinates(body_file_in, TopoBathy_flg, nlon, nlat)

    @info @sprintf """ Topography file: %s""" body_file_in

    FT = Float64 #NOTICE: I recommend the grid is always build in double precision even if the code is run in single 
    IT = Int64 #NOTICE: I recommend the grid is always build in double precision even if the code is run in single 

    open(body_file_in, "r") do ftopo_body
        @info @sprintf """ Grids.jl: Opening topography file ... DONE"""

        topo_body = readdlm(ftopo_body)
        (npoin_linear_grid, _) = size(topo_body)
        if npoin_linear_grid == 0
            error(" ERROR Grids.jl --> READTOPOtxt_xyz: grid file is empty!")
        end
        #
        # Create array
        #
        nnodes_lon, nnodes_lat, nnodes = nlon, nlat, nlon * nlat #Linear grid
        zTopo2d = Array{FT, 2}(undef, nnodes_lon, nnodes_lat)

        xTopo1d = Array{FT, 1}(undef, nnodes_lon)
        yTopo1d = Array{FT, 1}(undef, nnodes_lat)

        zTopo2d = reshape(topo_body[:, 3], nnodes_lon, nnodes_lat)
        #npoin = k

        #
        # Store 1D X array in the x direction: size -> nnodes_lat
        # and order in increasing order
        for i in 1:nnodes_lon
            xTopo1d[i] = topo_body[i, 1]
            #@info @sprintf """ X[%d]: %.16e""" i xTopo1d[i]
        end
        if minimum(xTopo1d) < 0
            xTopo1d = xTopo1d .- minimum(xTopo1d)
        end
        if xTopo1d[1] > xTopo1d[end]
            xTopo1d = reverse(xTopo1d)
        end

        #
        # Store 1D X array in the x direction: size -> nnodes_lat
        # and order in increasing order
        l = 0
        interval = nnodes_lon
        yTopo1d = topo_body[1:interval:nnodes, 2]

        #Shift if negative
        if minimum(yTopo1d) < 0
            yTopo1d = yTopo1d .- minimum(yTopo1d)
        end
        if yTopo1d[1] > yTopo1d[end]
            yTopo1d = reverse(yTopo1d)
        end

    end
    @info @sprintf """ Grids.jl: Closing topography file ... DONE"""

    return xTopo1d, yTopo1d, zTopo2d

end
#}}}


end
