##### Show boundary conditions

export show_bcs

bc_string(bc) = nameof(typeof(bc))
fmt_bcs(bcs) = "($(join(bcs, ", ")))"

function show_bcs(
    bl::BalanceLaw;
    include_module = false,
    table_complete = false,
)

    prog_vars = prognostic_vars(bl)
    if !isempty(prog_vars)
        all_bcs = boundary_conditions(bl)
        header = hcat(
            ["Equation"; "(Y_i)"],
            hcat(map(i -> ["BC"; "$i"], 1:length(all_bcs))...),
        )
        if include_module
            eqs = collect(string.(typeof.(prog_vars)))
        else
            eqs = collect(last.(split.(string.(typeof.(prog_vars)), ".")))
        end
        bcs_default = default_bcs(bl)
        bcs_entries = map(all_bcs) do bc
            bcs_used = used_bcs(bc)
            map(prognostic_vars(bl)) do prog
                fmt_bcs(map(bcs_used[prog]) do bc_pv
                    bc_string(bc_pv)
                end)
            end
        end
        table_complete || @warn "This BC table is temporarily incomplete"

        data = hcat(eqs, collect.(bcs_entries)...)
        pretty_table(
            data,
            header,
            header_crayon = crayon"yellow bold",
            subheader_crayon = crayon"green bold",
            crop = :none,
        )
    else
        msg = "Defining `prognostic_vars` and\n"
        msg *= "`default_bcs` for $(nameof(typeof(bl))) will\n"
        msg *= "enable printing a table of boundary conditions."
        @info msg
    end

end
