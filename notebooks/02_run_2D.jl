# 02 – Run a D-dimensional simulation
# This script is the config-driven equivalent of run_2D_clean.ipynb.
# It can run any of the four standard 2D configurations:
#   :thermal, :weibel_norel, :weibel_rel
# plus the 1D Landau-damping family:
#   :undamped_l, :undamped_s, :vlasov_exp_200, :vlasov_exp_40, :damped
# with any of the Maxwell solvers: :no_maxwell, :spectral, :sbp,
# :holbo (Holoborodko), :lanczos (Lanczos low-noise) and :weno (WENO-Z5).
# The spatial dimension D is inferred from the chosen configuration and kept
# constant throughout, so the core PIC loops are fully specialized.

using Pkg
const PROJECT_DIR = abspath(get(ENV, "PIC_PROJECT_DIR", normpath(joinpath(pwd(), ".."))))
isdir(joinpath(PROJECT_DIR, "src")) || error("PIC_PROJECT_DIR must point to the PIC_refactor project: $PROJECT_DIR")
Pkg.activate(PROJECT_DIR)
Pkg.instantiate()

cd(PROJECT_DIR)
include(joinpath(PROJECT_DIR, "src", "PIC.jl"))
using .PIC
ENV["GKSwstype"] = "nul"   # headless GR backend for terminal use
using Plots
using Statistics
using Base.Threads
using Printf
using FileIO, JLD2
using SummationByPartsOperators
println("project_dir = ", PROJECT_DIR)
println("nthreads = ", nthreads())

const order = 5
const factor = 2
#const D = 1

# -------------------------------------------------
# Choose configuration and solver
# -------------------------------------------------
# 2D options:   :thermal, :weibel_norel, :weibel_rel
# 1D options:   :undamped_l, :undamped_s, :vlasov_exp_200,
#               :vlasov_exp_40, :damped
#
config_name = :damped   # :thermal, :weibel_norel, :weibel_rel
solver_choice = :sbp     # :no_maxwell, :spectral, :sbp, :holbo, :lanczos, :weno

function load_config(name::Symbol)
    if name in (:thermal, :weibel_norel, :weibel_rel)
        return make_config(name)
    else
        return make_config_1d(name)
    end
end

cfg = load_config(config_name)
D = ndims(cfg.box)
N = cfg.N
J = cfg.J
box = cfg.box

println("Configuration: ", config_name)
println("Grid:          J = ", J)
println("Box:           ", box)
println("Data file:     ", cfg.data_name)

# Load initial particle distribution
file_name = joinpath(PROJECT_DIR, "Initial_Distributions", cfg.data_name * ".jld2")
isfile(file_name) || error("Initial data not found. Run 01_generate_initial_data.jl first: $file_name")

par_dis, run_name, pars, par_f_x, box_x, par_f_p, box_p = retrieve_initial_data_D(file_name)

# Build grid and state
g = PICGrid(box, J, order)
u = make_state(g, N)
u[1:(2D*N)] .= par_dis

# Deposit density, solve Poisson, set fields
n = zeros(size(g))
dep = DepositDensity(g, N)
deposit!(dep, u, g, n; factor=factor)
rho = n .- 1.0

E = zeros(D, size(g)...)
solve_poisson!(E, rho, g)
B = D >= 2 ? zeros(size(g)) : nothing
set_fields!(u, g, N, E, B)
coordinate_test(u, g, N)

err_max_initial, err_l2_initial = check_constraints(u, g, N, dep, factor)
println("initial  max |div E - rho| = ", err_max_initial)
println("initial  L2  |div E - rho| = ", err_l2_initial)

# Choose Maxwell solver
# Options: :no_maxwell, :spectral, :sbp, :holbo, :lanczos, :weno
derivative_order = 1
accuracy_order = 6
const stencil_width = 11   # used by the Holoborodko / Lanczos operators

if solver_choice == :no_maxwell
    maxwell = NoMaxwell{D}()
    suffix = "no_maxwell"
elseif solver_choice == :spectral
    maxwell = SpectralMaxwell(g)
    suffix = "spectral"
elseif solver_choice == :sbp
    Dops = ntuple(d -> periodic_derivative_operator(derivative_order=derivative_order,
                                                    accuracy_order=accuracy_order,
                                                    xmin=box.lo[d], xmax=box.hi[d], N=J[d]), D)
    maxwell = SBPMaxwell(Dops)
    suffix = "sbp"
elseif solver_choice == :holbo
    Dops = ntuple(d -> periodic_derivative_operator(Holoborodko2008(); derivative_order=1,
                                                    accuracy_order=4, stencil_width=stencil_width,
                                                    xmin=box.lo[d], xmax=box.hi[d], N=J[d]), D)
    maxwell = HolboMaxwell(Dops)
    suffix = "holbo"
elseif solver_choice == :lanczos
    Dops = ntuple(d -> periodic_derivative_operator(LanczosLowNoise(); derivative_order=1,
                                                    accuracy_order=4, stencil_width=stencil_width,
                                                    xmin=box.lo[d], xmax=box.hi[d], N=J[d]), D)
    maxwell = LanczosMaxwell(Dops)
    suffix = "lanczos"
elseif solver_choice == :weno
    maxwell = WENOMaxwell{D}()
    suffix = "weno"
else
    error("unknown solver_choice = $solver_choice (use :no_maxwell, :spectral, :sbp, :holbo, :lanczos, or :weno)")
end

p = RHSParams(g, N; factor=factor, maxwell=maxwell)
ws = StepWorkspace(u)

run_name_full = "$(cfg.run_name)_$(suffix)_o$(order)"
out_file = joinpath(PROJECT_DIR, "Results", run_name_full * ".jld2")
mkpath(dirname(out_file))
isfile(out_file) && rm(out_file)

println("Running with solver: ", solver_choice)
println("Output: ", out_file)

# Evolution parameters
t_i = 0.0
t_f = 0.10
M = 11
M_g = 10
dt = (t_f - t_i) / (M - 1)
@show dt, minimum(differentials(g)), dt / minimum(differentials(g))

# Save metadata and initial averages
save(out_file, Dict(
    "run_name" => run_name_full,
    "par_grid" => (N, J, box, order),
    "par_evolv" => (t_i, t_f, M, M_g, dt)
))
save_averages(out_file, 1, u, g, N; dep=dep, factor=factor)

# Time integration with RK4
let t = t_i
    out_every = div(M - 1, M_g - 1)
    for k in 2:M
        RK4_step!(rhs!, u, t, dt, p; ws=ws)
        t += dt
        if (k - 1) % out_every == 0
            j = (k - 1) ÷ out_every + 1
            save_averages(out_file, j, u, g, N; dep=dep, factor=factor)
            EK, EE = total_energy(u, g, N)
            err_max, err_l2 = check_constraints(u, g, N, dep, factor)
            println("j = ", j, "  t = ", round(t, digits=3),
                    "  E_total = ", EK + EE,
                    "  conts_max = ", err_max,
                    "  conts_L2 = ", err_l2)
        end
    end
end

err_max_final, err_l2_final = check_constraints(u, g, N, dep, factor)
println("final    max |div E - rho| = ", err_max_final)
println("final    L2  |div E - rho| = ", err_l2_final)
println("Done. Output: ", out_file)

# Final density snapshot
deposit!(dep, u, g, n; factor=factor)
if D == 1
    plot(range(box.lo[1], box.hi[1], length=length(n)), n,
         title="density at t = $(t_f)", xlabel="x", label=false)
elseif D == 2
    surface(n, title="density at t = $(t_f)")
end

# Weak constraint checks
p = 4
r0_frac = 0.1
N_test = 100
x0 = [box.lo .+ rand(length(box)) .* extent(box) for _ in 1:N_test]
if D == 2
    scatter([x[1] for x in x0], [x[2] for x in x0],
        xlabel="x", ylabel="y", label="x0", aspect_ratio=:equal)
end

weak_checks = [check_constraints_weak(u, g, N, dep, factor, x, r0_frac, p)[3] for x in x0]
@show sum(weak_checks) / length(weak_checks)

