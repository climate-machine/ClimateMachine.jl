using WriteVTK

function vtk_connectivity_map_highorder(Nq1, Nq2 = 1, Nq3 = 1)
    connectivity = Array{Int, 1}(undef, Nq1 * Nq2 * Nq3)
    L = LinearIndices((1:Nq1, 1:Nq2, 1:Nq3))

    corners = (
        (1, 1, 1),
        (Nq1, 1, 1),
        (Nq1, Nq2, 1),
        (1, Nq2, 1),
        (1, 1, Nq3),
        (Nq1, 1, Nq3),
        (Nq1, Nq2, Nq3),
        (1, Nq2, Nq3),
    )
    edges = (
        (2:(Nq1 - 1), 1:1, 1:1),
        (Nq1:Nq1, 2:(Nq2 - 1), 1:1),
        (2:(Nq1 - 1), Nq2:Nq2, 1:1),
        (1:1, 2:(Nq2 - 1), 1:1, 1:1),
        (2:(Nq1 - 1), 1:1, Nq3:Nq3),
        (Nq1:Nq1, 2:(Nq2 - 1), Nq3:Nq3),
        (2:(Nq1 - 1), Nq2:Nq2, Nq3:Nq3),
        (1:1, 2:(Nq2 - 1), Nq3:Nq3),
        (1:1, 1:1, 2:(Nq3 - 1)),
        (Nq1:Nq1, 1:1, 2:(Nq3 - 1)),
        (1:1, Nq2:Nq2, 2:(Nq3 - 1)),
        (Nq1:Nq1, Nq2:Nq2, 2:(Nq3 - 1)),
    )
    faces = (
        (1:1, 2:(Nq2 - 1), 2:(Nq3 - 1)),
        (Nq1:Nq1, 2:(Nq2 - 1), 2:(Nq3 - 1)),
        (2:(Nq1 - 1), 1:1, 2:(Nq3 - 1)),
        (2:(Nq1 - 1), Nq2:Nq2, 2:(Nq3 - 1)),
        (2:(Nq1 - 1), 2:(Nq2 - 1), 1:1),
        (2:(Nq1 - 1), 2:(Nq2 - 1), Nq3:Nq3),
    )
    if Nq2 == Nq3 == 1
        @assert Nq1 > 1
        corners = (corners[1:2]...,)
        edges = (edges[1],)
        faces = ()
    elseif Nq3 == 1
        @assert Nq1 > 1
        @assert Nq2 > 1
        corners = (corners[1:4]...,)
        edges = (edges[1:4]...,)
        faces = (faces[5],)
    end

    # corners
    iter = 1
    for (i, j, k) in corners
        connectivity[iter] = L[i, j, k]
        iter += 1
    end
    # edges
    for (is, js, ks) in edges
        for k in ks, j in js, i in is
            connectivity[iter] = L[i, j, k]
            iter += 1
        end
    end
    # faces
    for (is, js, ks) in faces
        for k in ks, j in js, i in is
            connectivity[iter] = L[i, j, k]
            iter += 1
        end
    end
    # interior
    for k in 2:(Nq3 - 1), j in 2:(Nq2 - 1), i in 2:(Nq1 - 1)
        connectivity[iter] = L[i, j, k]
        iter += 1
    end
    return connectivity
end

#=
This is the 1D WriteMesh routine
=#
function writemesh_highorder(
    base_name,
    x1;
    x2 = nothing,
    x3 = nothing,
    fields = (),
    realelems = 1:size(x1)[end],
)
    (Nq1, _) = size(x1)

    con = vtk_connectivity_map_highorder(Nq1)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, length(realelems))

    for (i, e) in enumerate(realelems)
        offset = (e - 1) * Nq1
        cells[i] = MeshCell(VTKCellTypes.VTK_LAGRANGE_CURVE, offset .+ con)
    end

    if x2 == nothing
        @assert isnothing(x3)
        vtkfile =
            vtk_grid("$(base_name)", @view(x1[:]), cells; compress = false)
    elseif x3 == nothing
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            cells;
            compress = false,
        )
    else
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            @view(x3[:]),
            cells;
            compress = false,
        )
    end
    length(realelems) > 0 && for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 2D WriteMesh routine
=#
function writemesh_highorder(
    base_name,
    x1,
    x2;
    x3 = nothing,
    fields = (),
    realelems = 1:size(x1)[end],
)
    @assert size(x1) == size(x2)
    (Nq1, Nq2, _) = size(x1)
    @assert Nq1 == Nq2
    con = vtk_connectivity_map_highorder(Nq1, Nq2)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, length(realelems))
    for (i, e) in enumerate(realelems)
        offset = (e - 1) * Nq1 * Nq2
        cells[i] =
            MeshCell(VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, offset .+ con[:])
    end

    if x3 == nothing
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            cells;
            compress = false,
        )
    else
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            @view(x3[:]),
            cells;
            compress = false,
        )
    end
    length(realelems) > 0 && for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 3D WriteMesh routine
=#
function writemesh_highorder(
    base_name,
    x1,
    x2,
    x3;
    fields = (),
    realelems = 1:size(x1)[end],
)
    (Nq1, Nq2, Nq3, _) = size(x1)
    @assert Nq1 == Nq2 == Nq3
    con = vtk_connectivity_map_highorder(Nq1, Nq2, Nq3)
    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, length(realelems))
    for (i, e) in enumerate(realelems)
        offset = (e - 1) * Nq1 * Nq2 * Nq3
        cells[i] = MeshCell(VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON, offset .+ con)
    end

    vtkfile = vtk_grid(
        "$(base_name)",
        @view(x1[:]),
        @view(x2[:]),
        @view(x3[:]),
        cells;
        compress = false,
    )
    length(realelems) > 0 && for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 1D WriteMesh routine
=#
function writemesh_raw(
    base_name,
    x1;
    x2 = nothing,
    x3 = nothing,
    fields = (),
    realelems = 1:size(x1)[end],
)
    (Nq1, _) = size(x1)
    Nsubcells = (Nq1 - 1)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, Nsubcells * length(realelems))
    for e in realelems
        offset = (e - 1) * Nq1
        for i in 1:(Nq1 - 1)
            cells[i + (e - 1) * Nsubcells] =
                MeshCell(VTKCellTypes.VTK_LINE, offset .+ [i, i + 1])
        end
    end

    vtkfile = vtk_grid("$(base_name)", @view(x1[:]), cells; compress = false)
    length(realelems) > 0 && for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 2D WriteMesh routine
=#
function writemesh_raw(
    base_name,
    x1,
    x2;
    x3 = nothing,
    fields = (),
    realelems = 1:size(x1)[end],
)
    @assert size(x1) == size(x2)
    (Nq1, Nq2, _) = size(x1)
    Nsubcells = (Nq1 - 1) * (Nq2 - 1)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, Nsubcells * length(realelems))
    ind = LinearIndices((1:Nq1, 1:Nq2))
    for e in realelems
        offset = (e - 1) * Nq1 * Nq2
        for j in 1:(Nq2 - 1)
            for i in 1:(Nq1 - 1)
                cells[i + (j - 1) * (Nq1 - 1) + (e - 1) * Nsubcells] = MeshCell(
                    VTKCellTypes.VTK_PIXEL,
                    offset .+ ind[i:(i + 1), j:(j + 1)][:],
                )
            end
        end
    end

    if x3 == nothing
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            cells;
            compress = false,
        )
    else
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            @view(x3[:]),
            cells;
            compress = false,
        )
    end
    length(realelems) > 0 && for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 3D WriteMesh routine
=#
function writemesh_raw(
    base_name,
    x1,
    x2,
    x3;
    fields = (),
    realelems = 1:size(x1)[end],
)
    (Nq1, Nq2, Nq3, _) = size(x1)
    (N1, N2, N3) = (Nq1 - 1, Nq2 - 1, Nq3 - 1)
    Nsubcells = N1 * N2 * N3

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, Nsubcells * length(realelems))
    ind = LinearIndices((1:Nq1, 1:Nq2, 1:Nq3))
    for e in realelems
        offset = (e - 1) * Nq1 * Nq2 * Nq3
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1
                    cells[i + (j - 1) * N1 + (k - 1) * N1 * N2 + (e - 1) * Nsubcells] =
                        MeshCell(
                            VTKCellTypes.VTK_VOXEL,
                            offset .+ ind[i:(i + 1), j:(j + 1), k:(k + 1)][:],
                        )
                end
            end
        end
    end

    vtkfile = vtk_grid(
        "$(base_name)",
        @view(x1[:]),
        @view(x2[:]),
        @view(x3[:]),
        cells;
        compress = false,
    )
    length(realelems) > 0 && for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end
