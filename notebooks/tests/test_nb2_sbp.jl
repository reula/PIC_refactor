cd(@__DIR__)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath("..","src","PIC.jl"))
using .PIC
using Base.Threads
using Printf
using FileIO, JLD2
using SummationByPartsOperators
println("nthreads = ", nthreads())

const N_exp = 5
const N = 10^N_exp
const order = 5
const factor = 2

J = (50, 50)
box = Box((0.0, 1.0, 0.0, 1.0))
g = PICGrid(box, J, order)

nm = [1, 1]
alpha_exp = 1
θ_exp = 3
θ = 10.0^(-θ_exp)

data_name = "par_dis_rel_thermal_nm_[$(nm[1]),$(nm[2])]_alp$(alpha_exp)_N$(N_exp)_Th$(θ_exp)"
file_name = joinpath("..", "Initial_Distributions", data_name * ".jld2")

par_dis, run_name, pars, par_f_x, box_x, par_f_p, box_p = retrieve_initial_data_D(file_name)

# short evolution for validation
t_i = 0.0
t_f = 0.1
M = 11
M_g = 3
dt = (t_f - t_i) / (M - 1)
@show dt

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
coordinate_test(u, g, N)

# SBP Maxwell solver (no dissipation)
Dx = periodic_derivative_operator(derivative_order=1, accuracy_order=6, xmin=box.lo[1], xmax=box.hi[1], N=J[1])
Dy = periodic_derivative_operator(derivative_order=1, accuracy_order=6, xmin=box.lo[2], xmax=box.hi[2], N=J[2])
maxwell_sbp = SBPMaxwell((Dx, Dy))

p = RHSParams(g, N; factor=factor, maxwell=maxwell_sbp)
ws = StepWorkspace(u)

run_name_full = "thermal_rel_sbp_J$(J[1])x$(J[2])_N$(N_exp)_Th$(θ_exp)_alp$(alpha_exp)_o$(order)_test"
out_file = joinpath("..", "Results", run_name_full * ".jld2")
mkpath(dirname(out_file))
isfile(out_file) && rm(out_file)

save(out_file, Dict(
    "run_name" => run_name_full,
    "par_grid" => (N, J, box, order),
    "par_evolv" => (t_i, t_f, M, M_g, dt)
))

save_averages(out_file, 1, u, g, N; dep=dep, factor=factor)

let t = t_i
    out_every = div(M - 1, M_g - 1)
    for k in 2:M
        RK4_step!(rhs!, u, t, dt, p; ws=ws)
        t += dt
        if (k - 1) % out_every == 0
            j = (k - 1) ÷ out_every + 1
            save_averages(out_file, j, u, g, N; dep=dep, factor=factor)
            EK, EE = total_energy(u, g, N)
            println("j = ", j, "  t = ", round(t, digits=3), "  E_total = ", EK + EE)
        end
    end
end
println("Done. Output: ", out_file)
