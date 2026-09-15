using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
cd(@__DIR__)
include(joinpath("..", "src", "PIC.jl"))
using .PIC

for name in (:thermal, :weibel_norel)
    cfg = make_config(name; N_exp=3)
    D = ndims(cfg.box)
    @assert D == 2
    N = cfg.N
    data_path = joinpath("..", "Initial_Distributions", "smoke_$(cfg.data_name)")

    par_dis = build_initial_data_D(
        data_path, N,
        cfg.f_x, cfg.f_x_max, cfg.par_f_x, cfg.box,
        cfg.f_p, cfg.f_p_max, cfg.par_f_p, cfg.box_p
    )

    r = reshape(par_dis, (2D, N))'
    pos = r[:, 1:D]
    mom = r[:, D+1:2D]

    g_dummy = PICGrid(cfg.box, cfg.J, 5)
    u_dummy = make_state(g_dummy, N)
    u_dummy[1:(2D*N)] .= par_dis

    println(name, ":")
    for d in 1:D
        println("  x$d range: ", extrema(pos[:, d]))
        println("  p$d range: ", extrema(mom[:, d]))
    end
    println("  T_rel = ", temperature(u_dummy, g_dummy, N; mode=:rel))
    println("  |P|/N = ", norm(total_momentum(u_dummy, g_dummy, N)) / N)
end