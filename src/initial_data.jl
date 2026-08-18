"""
Initial-data builders and distribution functions, ported from PIC-1D.
"""

using FileIO
using JLD2

# -----------------------------------------------------------------------------
# 1D/2D rejection sampling
# -----------------------------------------------------------------------------

function random_sampling_from_distribution(f, f_max, par_f, interval)
    x_min, x_max = interval
    fmax = f_max(par_f)
    while true
        x = x_min + (x_max - x_min) * rand()
        f_v = f(x, par_f)
        x_t = fmax * rand()
        if x_t <= f_v
            return x
        end
    end
end

function random_sampling_from_distribution_D(f, f_max, par_f, box::Box{D}) where {D}
    fmax = f_max(par_f)
    lo = box.lo
    hi = box.hi
    while true
        x = lo .+ (hi .- lo) .* rand(D)
        f_v = f(x, par_f)
        x_t = fmax * rand()
        if x_t <= f_v
            return x
        end
    end
end

# -----------------------------------------------------------------------------
# Momentum distributions
# -----------------------------------------------------------------------------

"""Relativistic thermal distribution (1D)."""
f_p_rel(p, (θ, norm); m2=1.0) = exp((m2 - sqrt(m2 + p^2)) / θ) / sqrt(θ * π * 2) / norm

"""Relativistic thermal distribution (D-dim)."""
f_p_rel(p::AbstractVector, (θ, norm); m2=1.0) = exp((m2 - sqrt(m2 + dot(p, p))) / θ) / sqrt(θ * π * 2)^length(p) / norm

"""Maximum of relativistic thermal distribution."""
f_p_rel_max((θ, norm, D); m2=1.0) = exp(m2) / sqrt(θ * π * 2)^D / norm

"""Non-relativistic thermal distribution."""
f_p_thermal(p, (θ,)) = exp(-p^2 / θ / 2) / sqrt(θ * π * 2)
f_p_thermal(p::AbstractVector, (θ, D)) = exp(-dot(p, p) / θ / 2) / sqrt(θ * π * 2)^D
f_p_thermal_max((θ, D)) = 1 / sqrt(θ * π * 2)^D

"""Non-relativistic Weibel anisotropic distribution."""
f_p_weibel_norel(p::AbstractVector, (θ, D, Ax)) = exp(-(dot(p, p) - p[1]^2 * (Ax / (1 + Ax))) / θ / 2) / sqrt(θ * π * 2)^D / sqrt(1 + Ax)
f_p_weibel_norel_max((θ, D, Ax)) = 1 / sqrt(θ * π * 2)^D / sqrt(1 + Ax)

"""Two-counter-stream relativistic distribution."""
function f_p_two_particle_distribution(p::AbstractVector, par_f_p; m=1.0)
    θ1, θ2, v::AbstractVector, norm, D = par_f_p
    γ = 1 / sqrt(1 - dot(v, v))
    return exp((m - sqrt(m^2 + dot(p, p))) / θ1) * exp((m - sqrt(m^2 + dot(p, p)) - dot(v, p)) / θ2 * γ) / norm
end

function f_p_two_particle_distribution_max(par_f_p; m=1.0)
    θ1, θ2, v::AbstractVector, norm, D = par_f_p
    γ = 1 / sqrt(1 - dot(v, v))
    return exp(-m / θ2 * (1 - 1 / γ)) / norm
end

# -----------------------------------------------------------------------------
# Space distributions
# -----------------------------------------------------------------------------

"""1D space perturbation: (1 + α cos(k x))/L."""
function f_x(x::Real, par_f_x)
    α, mn, L = par_f_x
    k = 2π * mn / L
    return (1 + α * cos(k * x)) / L
end

function f_x_max(par_f_x)
    α, mn, L = par_f_x
    return (1 + abs(α)) / L
end

"""D-dimensional space perturbation."""
function f_x(x::AbstractVector, par_f_x)
    α, m, box = par_f_x
    L = extent(box)
    k = 2π .* m ./ L
    return (1 + α * cos(dot(k, x))) / volume(box)
end

function f_x_max(par_f_x::Tuple{Any,Any,Box{D}}) where {D}
    α, m, box = par_f_x
    return (1 + abs(α)) / volume(box)
end

# -----------------------------------------------------------------------------
# Initial-data builders
# -----------------------------------------------------------------------------

"""
    build_initial_data_D(data_name, N, f_x, f_x_max, par_f_x, box_x,
                         f_p, f_p_max, par_f_p, box_p; symmetric=true)

Build a D-dimensional initial particle distribution and save it to
`Initial_Distributions/<data_name>.jld2`.
Returns the flat phase-space vector `par_dis` compatible with `make_state`.
"""
function build_initial_data_D(data_name::String, N::Int,
                              f_x, f_x_max, par_f_x, box_x::Box{D},
                              f_p, f_p_max, par_f_p, box_p::Box{D};
                              symmetric::Bool=true) where {D}
    par_dis = zeros(2D * N)
    @inline rx(i) = (i - 1) * 2D .+ (1:D)
    @inline rp(i) = (i - 1) * 2D .+ (D + 1:2D)
    # positions
    for i in 1:N
        par_dis[rx(i)] = random_sampling_from_distribution_D(f_x, f_x_max, par_f_x, box_x)
    end
    # momenta
    if symmetric
        iseven(N) || error("symmetric momentum option requires even N")
        half = N ÷ 2
        for i in 1:half
            pm = random_sampling_from_distribution_D(f_p, f_p_max, par_f_p, box_p)
            par_dis[rp(i)] = pm
            j = i + half
            par_dis[rp(j)] = -pm
        end
    else
        for i in 1:N
            pm = random_sampling_from_distribution_D(f_p, f_p_max, par_f_p, box_p)
            par_dis[rp(i)] = pm
        end
    end

    mkpath(dirname(data_name))
    file_name = data_name * ".jld2"
    run_pars = Dict("data_name" => data_name, "pars" => (N,),
                    "par_f_x" => par_f_x, "Box_x" => box_x,
                    "par_f_p" => par_f_p, "Box_p" => box_p)
    save(file_name, run_pars)
    jldopen(file_name, "a+") do file
        file["par_dis"] = par_dis
    end
    return par_dis
end

"""
    retrieve_initial_data_D(file_name)

Load a distribution created by `build_initial_data_D`.
"""
function retrieve_initial_data_D(file_name::String)
    data = load(file_name)
    run_name = data["data_name"]
    pars = data["pars"]
    par_f_x = data["par_f_x"]
    box_x = data["Box_x"]
    par_f_p = data["par_f_p"]
    box_p = data["Box_p"]
    return data["par_dis"], run_name, pars, par_f_x, box_x, par_f_p, box_p
end
