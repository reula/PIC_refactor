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

"""Maximum of relativistic thermal distribution (D-dim)."""
f_p_rel_max((θ, norm, D); m2=1.0) = exp(m2) / sqrt(θ * π * 2)^D / norm

"""Maximum of relativistic thermal distribution (1D)."""
f_p_rel_max((θ, norm)::Tuple{<:Real,<:Real}; m2=1.0) = exp(m2) / sqrt(θ * π * 2) / norm

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

"""1D space perturbation: (1 + α cos(k x))/L.

Accepts the scalar parameter form `par_f_x = (α, m, L)` with scalar `m`/`L`
as well as the box form `(α, [m], box)` used by the 1D configs, so that it
stays consistent with `f_x(x::AbstractVector, par_f_x)`.
"""
function f_x(x::Real, par_f_x)
    α, mn, L = par_f_x
    Lv = L isa Box ? extent(L, 1) : L
    m = mn isa AbstractVector ? mn[1] : mn
    k = 2π * m / Lv
    return (1 + α * cos(k * x)) / Lv
end

"""Maximum of the 1D space perturbation (scalar-length form)."""
function f_x_max(par_f_x)
    α, mn, L = par_f_x
    Lv = L isa Box ? extent(L, 1) : L
    return (1 + abs(α)) / Lv
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
    build_initial_data_1D(data_name, N, f_x, f_x_max, par_f_x, box_x,
                          f_p, f_p_max, par_f_p, box_p; symmetric=true)

Build a 1D initial particle distribution and save it to
`Initial_Distributions/<data_name>.jld2`.
Returns the flat phase-space vector `par_dis` compatible with `make_state`.
"""
function build_initial_data_1D(data_name::String, N::Int,
                               f_x, f_x_max, par_f_x, box_x::Box{1},
                               f_p, f_p_max, par_f_p, box_p::Box{1};
                               symmetric::Bool=true)
    par_dis = zeros(2N)
    interval_x = (box_x.lo[1], box_x.hi[1])
    interval_p = (box_p.lo[1], box_p.hi[1])
    # interleaved layout: [x1, p1, x2, p2, ...], matching make_state
    @inline rx(i) = 2(i - 1) + 1
    @inline rp(i) = 2(i - 1) + 2
    # positions
    for i in 1:N
        par_dis[rx(i)] = random_sampling_from_distribution(f_x, f_x_max, par_f_x, interval_x)
    end
    # momenta
    if symmetric
        iseven(N) || error("symmetric momentum option requires even N")
        half = N ÷ 2
        for i in 1:half
            pm = random_sampling_from_distribution(f_p, f_p_max, par_f_p, interval_p)
            par_dis[rp(i)] = pm
            j = i + half
            par_dis[rp(j)] = -pm
        end
    else
        for i in 1:N
            par_dis[rp(i)] = random_sampling_from_distribution(f_p, f_p_max, par_f_p, interval_p)
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
    retrieve_initial_data_1D(file_name)

Load a distribution created by `build_initial_data_1D`.
"""
function retrieve_initial_data_1D(file_name::String)
    data = load(file_name)
    run_name = data["data_name"]
    pars = data["pars"]
    par_f_x = data["par_f_x"]
    box_x = data["Box_x"]
    par_f_p = data["par_f_p"]
    box_p = data["Box_p"]
    return data["par_dis"], run_name, pars, par_f_x, box_x, par_f_p, box_p
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

# ---------------------------------------------------------------------------
# Midpoint integration helper (ported from PIC-1D)
# ---------------------------------------------------------------------------

"""
    int_mid_point_f(f, par, n, box::Box{D})

Midpoint-rule integral of `f(x, par)` over `box` using `n[d]` points per
dimension.
"""
function int_mid_point_f(f, par, n::AbstractVector{<:Integer}, box::Box{D}) where {D}
    D == length(n) || error("length(n) must match dimension D=$D")
    lo = box.lo
    hi = box.hi
    if D == 1
        dp = (hi[1] - lo[1]) / (n[1] - 1)
        pp = [lo[1] + dp * (i - 1) for i in 1:n[1]]
        F = [f(p, par) for p in pp]
        return (sum(F) - 0.5 * (F[1] + F[end])) * dp
    elseif D == 2
        dp1 = (hi[1] - lo[1]) / (n[1] - 1)
        dp2 = (hi[2] - lo[2]) / (n[2] - 1)
        pp1 = [lo[1] + dp1 * (i - 1) for i in 1:n[1]]
        pp2 = [lo[2] + dp2 * (i - 1) for i in 1:n[2]]
        F = [f([p1, p2], par) for p1 in pp1, p2 in pp2]
        return (sum(F)
                - 0.5 * (sum(F[1, :] + F[end, :]) + sum(F[:, 1] + F[:, end]))
                + 0.25 * (F[1, 1] + F[1, end] + F[end, 1] + F[end, end])) * dp1 * dp2
    elseif D == 3
        dp1 = (hi[1] - lo[1]) / (n[1] - 1)
        dp2 = (hi[2] - lo[2]) / (n[2] - 1)
        dp3 = (hi[3] - lo[3]) / (n[3] - 1)
        pp1 = [lo[1] + dp1 * (i - 1) for i in 1:n[1]]
        pp2 = [lo[2] + dp2 * (i - 1) for i in 1:n[2]]
        pp3 = [lo[3] + dp3 * (i - 1) for i in 1:n[3]]
        F = [f([p1, p2, p3], par) for p1 in pp1, p2 in pp2, p3 in pp3]
        Int = sum(F)
        Int -= 0.5 * (sum(F[1, :, :] + F[end, :, :])
                    + sum(F[:, 1, :] + F[:, end, :])
                    + sum(F[:, :, 1] + F[:, :, end]))
        Int += 0.25 * sum(F[1, 1, :] + F[1, end, :] + F[end, 1, :] + F[end, end, :])
        Int += 0.25 * sum(F[1, :, 1] + F[1, :, end] + F[end, :, 1] + F[end, :, end])
        Int += 0.25 * sum(F[:, 1, 1] + F[:, 1, end] + F[:, end, 1] + F[:, end, end])
        Int -= 0.125 * (F[1, 1, 1] + F[1, 1, end] + F[1, end, 1] + F[1, end, end]
                      + F[end, 1, 1] + F[end, 1, end] + F[end, end, 1] + F[end, end, end])
        return Int * dp1 * dp2 * dp3
    else
        error("int_mid_point_f only implemented for D = 1, 2, 3")
    end
end
