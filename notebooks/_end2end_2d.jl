using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Printf
using SummationByPartsOperators
cd(@__DIR__)
include(joinpath("..", "src", "PIC.jl"))
using .PIC

config_name = :thermal
N_exp = 3
cfg = make_config(config_name; N_exp=N_exp)
D = ndims(cfg.box)
N = cfg.N
J = cfg.J
box = cfg.box
@assert D == 2

println("=== generating initial data ===")
data_path = joinpath("..", "Initial_Distributions", cfg.data_name)
par_dis = build_initial_data_D(
    data_path, N,
    cfg.f_x, cfg.f_x_max, cfg.par_f_x, cfg.box,
    cfg.f_p, cfg.f_p_max, cfg.par_f_p, cfg.box_p
)

for solver_choice in (:no_maxwell, :spectral, :sbp, :holbo, :lanczos, :weno)
    println("\n=== running with solver: ", solver_choice, " ===")

    g = PICGrid(box, J, 5)
    u = make_state(g, N)
    u[1:(2D*N)] .= par_dis

    n = zeros(size(g))
    dep = DepositDensity(g, N)
    deposit!(dep, u, g, n; factor=2)
    rho = n .- 1.0

    E = zeros(D, size(g)...)
    solve_poisson!(E, rho, g)
    B = zeros(size(g))
    set_fields!(u, g, N, E, B)
    coordinate_test(u, g, N)

    err_max_initial, err_l2_initial = check_constraints(u, g, N, dep, 2)
    println("initial max |div E - rho| = ", err_max_initial)

    if solver_choice == :no_maxwell
        maxwell = NoMaxwell{D}()
    elseif solver_choice == :spectral
        maxwell = SpectralMaxwell(g)
    elseif solver_choice == :sbp
        Dops = ntuple(d -> periodic_derivative_operator(
            derivative_order=1, accuracy_order=6,
            xmin=box.lo[d], xmax=box.hi[d], N=J[d]), D)
        maxwell = SBPMaxwell(Dops)
    elseif solver_choice == :holbo
        Dops = ntuple(d -> periodic_derivative_operator(Holoborodko2008();
            derivative_order=1, accuracy_order=4, stencil_width=11,
            xmin=box.lo[d], xmax=box.hi[d], N=J[d]), D)
        maxwell = HolboMaxwell(Dops)
    elseif solver_choice == :lanczos
        Dops = ntuple(d -> periodic_derivative_operator(LanczosLowNoise();
            derivative_order=1, accuracy_order=4, stencil_width=11,
            xmin=box.lo[d], xmax=box.hi[d], N=J[d]), D)
        maxwell = LanczosMaxwell(Dops)
    elseif solver_choice == :weno
        maxwell = WENOMaxwell{D}()
    end

    p = RHSParams(g, N; factor=2, maxwell=maxwell)
    ws = StepWorkspace(u)

    t_i = 0.0
    t_f = 0.05
    M = 6
    dt = (t_f - t_i) / (M - 1)
    println("dt = ", dt)

    t = t_i
    for k in 2:M
        RK4_step!(rhs!, u, t, dt, p; ws=ws)
        t += dt
        EK, EE = total_energy(u, g, N)
        err_max, err_l2 = check_constraints(u, g, N, dep, 2)
        println(@sprintf "k=%d t=%.4f E_total=%.6e cont_max=%.3e cont_L2=%.3e" k t EK+EE err_max err_l2)
    end
end

println("\n=== 2D end-to-end test passed ===")