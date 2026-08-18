"""
Right-hand side assembly and time stepping.
"""

mutable struct RHSWorkspace{D}
    r::Matrix{Float64}
    v::Matrix{Float64}
    idx::Matrix{Int}
    y::Matrix{Float64}
    S::Array{Float64}
    Fi::Matrix{Float64}
    du::Vector{Float64}
    function RHSWorkspace(g::PICGrid{D}, N::Int) where {D}
        new{D}(
            zeros(Float64, N, D),
            zeros(Float64, N, D),
            ones(Int, N, D),
            zeros(Float64, N, D),
            zeros(Float64, D, size(g)...),
            zeros(Float64, N, D),
            zeros(Float64, state_length(g, N))
        )
    end
end

struct RHSParams{Grid<:PICGrid,MW<:MaxwellSolver,Depo<:DepositCurrent}
    grid::Grid
    N::Int
    factor::Int
    maxwell::MW
    deposit::Depo
    workspace::RHSWorkspace
end

function RHSParams(g::PICGrid{D}, N::Int; factor::Int=1, maxwell::MaxwellSolver{D}=NoMaxwell{D}()) where {D}
    dep = DepositCurrent(g, N)
    ws = RHSWorkspace(g, N)
    return RHSParams(g, N, factor, maxwell, dep, ws)
end

function add_current_source!(dF::AbstractArray{Float64}, S::AbstractArray{Float64}, g::PICGrid{D}) where {D}
    @inbounds for I in CartesianIndices(size(g))
        idx = I.I
        for d in 1:D
            dF[d, idx...] += S[d, idx...]
        end
    end
    return dF
end

"""
    rhs!(du, u, t, p)

Compute the full RHS in-place.
"""
function rhs!(du::AbstractVector, u::AbstractVector, t::Real, p::RHSParams)
    g = p.grid
    N = p.N
    D = ndims(g)
    ws = p.workspace

    make_periodic!(u, g, N)

    positions!(ws.r, u, g)
    velocities!(ws.v, u, g)
    get_indices_and_y!(ws.idx, ws.y, ws.r, g)

    # Deposit current
    deposit!(p.deposit, u, g, ws.S; factor=p.factor)

    # Zero RHS and views
    fill!(du, 0.0)
    F = reshape(view(u, particle_dof(g, N) + 1:length(u)), (field_components(g), size(g)...))
    dF = reshape(view(du, particle_dof(g, N) + 1:length(du)), (field_components(g), size(g)...))

    # Maxwell part
    if !(p.maxwell isa NoMaxwell)
        add_maxwell_rhs!(dF, F, g, p.maxwell)
    end

    # Current source (Ampere)
    add_current_source!(dF, ws.S, g)

    # Interpolate Lorentz force
    E = get_E(u, g, N)
    B = D ≥ 2 ? get_B(u, g, N) : nothing
    interpolate_field!(ws.Fi, E, B, ws.idx, ws.y, ws.v, g; factor=p.factor)

    # Particle RHS
    @threads :static for i in 1:N
        @inbounds for d in 1:D
            du[(i - 1) * 2D + d] = ws.v[i, d]
            du[(i - 1) * 2D + D + d] = -ws.Fi[i, d]
        end
    end

    return du
end

# -----------------------------------------------------------------------------
# Time steppers
# -----------------------------------------------------------------------------

function RK4_step!(rhs!::Function, y::AbstractVector, t::Real, dt::Real, p; ws=nothing)
    if ws === nothing
        k1 = similar(y)
        k2 = similar(y)
        k3 = similar(y)
        k4 = similar(y)
        tmp = similar(y)
    else
        k1, k2, k3, k4, tmp = ws.k1, ws.k2, ws.k3, ws.k4, ws.tmp
    end
    rhs!(k1, y, t, p)
    @. tmp = y + 0.5 * dt * k1
    rhs!(k2, tmp, t + 0.5 * dt, p)
    @. tmp = y + 0.5 * dt * k2
    rhs!(k3, tmp, t + 0.5 * dt, p)
    @. tmp = y + dt * k3
    rhs!(k4, tmp, t + dt, p)
    @. y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y
end

"""
    StepWorkspace

Pre-allocated temporaries for RK4_step! to avoid allocations in the time loop.
"""
mutable struct StepWorkspace
    k1::Vector{Float64}
    k2::Vector{Float64}
    k3::Vector{Float64}
    k4::Vector{Float64}
    tmp::Vector{Float64}
    function StepWorkspace(y::AbstractVector)
        new(similar(y), similar(y), similar(y), similar(y), similar(y))
    end
end
