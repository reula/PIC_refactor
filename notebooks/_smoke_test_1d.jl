using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
cd(@__DIR__)
include(joinpath("..", "src", "PIC.jl"))
using .PIC
using LinearAlgebra

for name in (:undamped_l, :undamped_s, :vlasov_exp_200, :vlasov_exp_40, :damped)
    cfg = make_config_1d(name)
    D = ndims(cfg.box)
    @assert D == 1
    N = 8 * 10^2  # small for speed
    data_path = joinpath("..", "Initial_Distributions", "smoke_$(cfg.data_name)")

    par_dis = build_initial_data_D(
        data_path,
        N,
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
    println("  x range: ", extrema(pos[:, 1]))
    println("  p range: ", extrema(mom[:, 1]))
    println("  T_rel = ", temperature(u_dummy, g_dummy, N; mode=:rel))
    println("  T_norel = ", temperature(u_dummy, g_dummy, N; mode=:norel))
    println("  |P|/N = ", norm(total_momentum(u_dummy, g_dummy, N)) / N)
end