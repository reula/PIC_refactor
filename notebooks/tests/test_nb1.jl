cd(@__DIR__)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath("..","src","PIC.jl"))
using .PIC
using Plots
using Statistics
using FFTW
using LinearAlgebra

const N_exp = 5
const N = 10^N_exp
const order = 5
const factor = 2

J = (50, 50)
box = Box((0.0, 1.0, 0.0, 1.0))
nm = [1, 1]
alpha_exp = 1
α = 10.0^(-alpha_exp)
θ_exp = 3
θ = 10.0^(-θ_exp)
p_max = sqrt((1 + 10θ)^2 - 1)
box_p = Box((-p_max, p_max, -p_max, p_max))

data_name = "par_dis_rel_thermal_nm_[$(nm[1]),$(nm[2])]_alp$(alpha_exp)_N$(N_exp)_Th$(θ_exp)"
par_f_x = (α, nm, box)
par_f_p = (θ, 2)

par_dis = build_initial_data_D(
    joinpath("..", "Initial_Distributions", data_name),
    N,
    f_x, f_x_max, par_f_x, box,
    f_p_thermal, f_p_thermal_max, par_f_p, box_p
)

println("Saved to: Initial_Distributions/$(data_name).jld2")
println("length(par_dis) = ", length(par_dis))

# Reshape into N×4: [x, y, px, py]
r = reshape(par_dis, (4, N))'
x = r[:, 1]
y = r[:, 2]
px = r[:, 3]
py = r[:, 4]

# Build a minimal state for module diagnostics
g_dummy = PICGrid(box, J, order)
u_dummy = make_state(g_dummy, N)
u_dummy[1:4N] .= par_dis

# Temperatures
T_rel = temperature(u_dummy, g_dummy, N; mode=:rel)
T_norel = temperature(u_dummy, g_dummy, N; mode=:norel)
println("relativistic temperature T = ", T_rel, "  (target θ = ", θ, ")")
println("non-relativistic temperature T = ", T_norel, "  (target θ = ", θ, ")")

# Total momentum (should be ~0 with symmetric pairs)
P_total = total_momentum(u_dummy, g_dummy, N)
println("total momentum P = ", P_total)
println("|P|/N = ", norm(P_total) / N)

println("x range: ", extrema(x))
println("y range: ", extrema(y))
println("<px^2> = ", mean(px.^2), "  (target θ = ", θ, ")")
println("<py^2> = ", mean(py.^2), "  (target θ = ", θ, ")")

println("Done")
