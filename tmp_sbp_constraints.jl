include(joinpath(@__DIR__, "src", "PIC.jl"))
using .PIC
using SummationByPartsOperators

function main()
    N_exp = 5
    N = 10^N_exp
    order = 5
    factor = 2
    J = (50, 50)
    box = Box((0.0, 1.0, 0.0, 1.0))
    g = PICGrid(box, J, order)

    nm = [1, 1]
    alpha_exp = 1
    θ_exp = 3
    data_name = "par_dis_rel_thermal_nm_[$(nm[1]),$(nm[2])]_alp$(alpha_exp)_N$(N_exp)_Th$(θ_exp)"
    file_name = joinpath("Initial_Distributions", data_name * ".jld2")
    par_dis, _, _, _, _, _, _ = retrieve_initial_data_D(file_name)

    u = make_state(g, N)
    u[1:4N] .= par_dis
    n = zeros(size(g))
    dep = DepositDensity(g, N)
    deposit!(dep, u, g, n; factor=factor)
    rho = n .- 1.0
    E = zeros(2, size(g)...)
    solve_poisson!(E, rho, g)
    B = zeros(size(g))
    set_fields!(u, g, N, E, B)

    err_max_initial, err_l2_initial = check_constraints(u, g, N, dep, factor)
    println("initial  max |div E - rho| = ", err_max_initial)
    println("initial  L2  |div E - rho| = ", err_l2_initial)

    Dx = periodic_derivative_operator(derivative_order=1, accuracy_order=6, xmin=box.lo[1], xmax=box.hi[1], N=J[1])
    Dy = periodic_derivative_operator(derivative_order=1, accuracy_order=6, xmin=box.lo[2], xmax=box.hi[2], N=J[2])
    maxwell_sbp = SBPMaxwell((Dx, Dy))
    p = RHSParams(g, N; factor=factor, maxwell=maxwell_sbp)
    ws = StepWorkspace(u)

    dt = 0.01
    M = 11
    t_sim = 0.0
    for k = 2:M
        RK4_step!(rhs!, u, t_sim, dt, p; ws=ws)
        t_sim += dt
    end
    EK, EE = total_energy(u, g, N)
    println("final E_total = ", EK + EE)
    err_max_final, err_l2_final = check_constraints(u, g, N, dep, factor)
    println("final  max |div E - rho| = ", err_max_final)
    println("final  L2  |div E - rho| = ", err_l2_final)
end

main()
