# 01 – Generate initial data
# This script generates particle initial conditions for the refactored PIC code.
# It supports the standard 2D configurations (thermal, Weibel) plus
# the 1D Landau-damping family from ini_dat_v2.ipynb (including the damped case).
#
# Pick one configuration below and run the script.  The dimension D is inferred
# from the chosen config and remains a compile-time constant for the core PIC
# routines.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

cd(@__DIR__)
include(joinpath("..", "src", "PIC.jl"))
using .PIC
ENV["GKSwstype"] = "nul"   # headless GR backend for terminal use
using Plots
using Statistics
using FFTW
using LinearAlgebra

const order = 5 #shape smoothness
const factor = 2

# -------------------------------------------------
# Choose configuration
# -------------------------------------------------
# 2D options:   :thermal, :weibel_norel, :weibel_rel
# 1D options:   :undamped_l, :undamped_s, :vlasov_exp_200,
#               :vlasov_exp_40, :damped
#
config_name = :thermal

function load_config(name::Symbol)
    if name in (:thermal, :weibel_norel, :weibel_rel)
        return make_config(name)
    else
        return make_config_1d(name)
    end
end

cfg = load_config(config_name)
D = ndims(cfg.box)

@show cfg.name
@show cfg.N
@show cfg.J
@show cfg.box
@show cfg.data_name

# Generate and save the particle distribution using the D-generic builder.
# This produces the per-particle interleaved layout
#   [x1, ..., xD, p1, ..., pD, x1, ..., xD, p1, ..., pD, ...]
# that make_state expects, regardless of dimension.
data_path = joinpath("..", "Initial_Distributions", cfg.data_name)

par_dis = build_initial_data_D(
    data_path,
    cfg.N,
    cfg.f_x, cfg.f_x_max, cfg.par_f_x, cfg.box,
    cfg.f_p, cfg.f_p_max, cfg.par_f_p, cfg.box_p
)

println("Saved to: Initial_Distributions/$(cfg.data_name).jld2")
println("length(par_dis) = ", length(par_dis))

# Reshape into N x 2D: [x1..xD, p1..pD] per particle
N = cfg.N
r = reshape(par_dis, (2D, N))'
pos = r[:, 1:D]
mom = r[:, D+1:2D]

# Minimal state for module diagnostics
g_dummy = PICGrid(cfg.box, cfg.J, order)
u_dummy = make_state(g_dummy, N)
u_dummy[1:(2D*N)] .= par_dis

T_rel   = temperature(u_dummy, g_dummy, N; mode=:rel)
T_norel = temperature(u_dummy, g_dummy, N; mode=:norel)
println("relativistic temperature T = ", T_rel, "  (target θ = ", cfg.theta, ")")
println("non-relativistic temperature T = ", T_norel, "  (target θ = ", cfg.theta, ")")

P_total = total_momentum(u_dummy, g_dummy, N)
println("total momentum P = ", P_total)
println("|P|/N = ", norm(P_total) / N)

for d in 1:D
    println("x$(d) range: ", extrema(pos[:, d]))
    println("p$(d) range: ", extrema(mom[:, d]))
    println("<p$(d)^2> = ", mean(mom[:, d].^2), "  (target θ = ", cfg.theta, ")")
end

# -------------------------------------------------
# Spatial plots
# -------------------------------------------------
if D == 1
    x = pos[:, 1]
    hx = histogram(x, bins=50, normalize=true, label="sampled", title="density profile")
    kx = 2π * cfg.m / cfg.L
    analytic_x = xx -> (1 + cfg.alpha * cos(kx * xx)) / cfg.L
    xx = range(cfg.box.lo[1], cfg.box.hi[1], length=200)
    plot!(hx, xx, analytic_x.(xx), lw=2, label="analytic")
    display(hx)
elseif D == 2
    x = pos[:, 1]
    y = pos[:, 2]
    h2 = histogram2d(x, y, bins=50,
        title="spatial distribution", c=:thermal, aspectratio=1,
        xlabel="x", ylabel="y")

    hx = histogram(x, bins=50, normalize=true, label="sampled", title="density profile along x")
    kx = 2π * cfg.nm[1] / extent(cfg.box, 1)
    analytic_x = xx -> (1 + cfg.alpha * cos(kx * xx)) / extent(cfg.box, 1)
    xx = range(cfg.box.lo[1], cfg.box.hi[1], length=200)
    plot!(hx, xx, analytic_x.(xx), lw=2, label="analytic")

    hy = histogram(y, bins=50, normalize=true, label="sampled", title="density profile along y")
    ky = 2π * cfg.nm[2] / extent(cfg.box, 2)
    analytic_y = yy -> (1 + cfg.alpha * cos(ky * yy)) / extent(cfg.box, 2)
    yy = range(cfg.box.lo[2], cfg.box.hi[2], length=200)
    plot!(hy, yy, analytic_y.(yy), lw=2, label="analytic")

plot(h2, hx, hy, layout=(1, 3), size=(1200, 350))
    display(plot!(dpi=100))
end

# -------------------------------------------------
# Momentum plots
# -------------------------------------------------
θ = cfg.theta
p_max = cfg.box_p.hi[1]
pp = range(-p_max, p_max, length=200)

if D == 1
    px = mom[:, 1]
    norm_p = cfg.par_f_p[2]
    hpx = histogram(px, bins=50, normalize=true, label="sampled", title="momentum distribution")
    analytic_p = pp -> exp((1 - sqrt(1 + pp^2)) / θ) / sqrt(θ * π * 2) / norm_p
    plot!(hpx, pp, analytic_p.(pp), lw=2, label="analytic")
    display(hpx)
elseif D == 2
    px = mom[:, 1]
    py = mom[:, 2]
    mp2 = histogram2d(px, py, bins=50,
        title="momentum distribution", c=:thermal, aspectratio=1,
        xlabel="px", ylabel="py")

    analytic_p = pp -> exp(-pp.^2 / θ / 2) / sqrt(θ * π * 2)

    hpx = histogram(px, bins=50, normalize=true, label="sampled", title="px distribution")
    plot!(hpx, pp, analytic_p.(pp), lw=2, label="analytic")

    hpy = histogram(py, bins=50, normalize=true, label="sampled", title="py distribution")
    plot!(hpy, pp, analytic_p.(pp), lw=2, label="analytic")

    plot(mp2, hpx, hpy, layout=(1, 3), size=(1200, 350))
    display(plot!(dpi=100))
end

# -------------------------------------------------
# Scatter plots of the first few thousand particles
# -------------------------------------------------
nplot = min(N, 4000)
if D == 1
    p1 = scatter(1:nplot, pos[1:nplot, 1],
        title="positions (first $nplot particles)",
        xlabel="particle index", ylabel="x", markersize=1, label=false)
    p2 = scatter(1:nplot, mom[1:nplot, 1],
        title="momenta (first $nplot particles)",
        xlabel="particle index", ylabel="px", markersize=1, label=false)
    plot(p1, p2, layout=(1, 2), size=(900, 350))
    display(plot!(dpi=100))
elseif D == 2
    p1 = scatter(pos[1:nplot, 1], pos[1:nplot, 2],
        title="positions (first $nplot particles)", aspectratio=1,
        xlabel="x", ylabel="y", markersize=1, label=false)
    p2 = scatter(mom[1:nplot, 1], mom[1:nplot, 2],
        title="momenta (first $nplot particles)", aspectratio=1,
        xlabel="px", ylabel="py", markersize=1, label=false)
    plot(p1, p2, layout=(1, 2), size=(900, 350))
    display(plot!(dpi=100))
end

