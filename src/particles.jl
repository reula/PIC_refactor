"""Relativistic momentum -> velocity."""
@inline p2v(p::SVector{D,T}; m::T=T(1.0)) where {D,T} = p / sqrt(m^2 + dot(p, p))
@inline p2v(p::AbstractVector{T}; m::Real=1.0) where {T} = p / sqrt(m^2 + dot(p, p))

"""Relativistic velocity -> momentum."""
@inline function v2p(v::SVector{D,T}; m::T=T(1.0)) where {D,T}
    γv = T(1.0) / sqrt(T(1.0) - dot(v, v))
    return m * γv * v
end

"""Lorentz factor from momentum."""
@inline γ_p(p::SVector{D,T}; m::T=T(1.0)) where {D,T} = sqrt(m^2 + dot(p, p)) / m

"""Get particle position as SVector (copy)."""
@inline function get_positions(u::AbstractVector, i::Int, ::PICGrid{D}) where {D}
    base = (i - 1) * 2D
    SVector{D}(u[base + d] for d in 1:D)
end

"""Get particle momentum as SVector (copy)."""
@inline function get_momenta(u::AbstractVector, i::Int, ::PICGrid{D}) where {D}
    base = (i - 1) * 2D
    SVector{D}(u[base + D + d] for d in 1:D)
end

"""
    positions!(r, u, g)

Fill N×D matrix `r` with particle positions from state vector `u`.
"""
function positions!(r::Matrix{Float64}, u::AbstractVector, g::PICGrid{D}) where {D}
    N = size(r, 1)
    @threads :static for i in 1:N
        @inbounds for d in 1:D
            r[i, d] = u[(i - 1) * 2D + d]
        end
    end
    return r
end

"""
    velocities!(v, u, g; m=1.0)

Fill N×D matrix `v` with relativistic velocities computed from momenta.
"""
function velocities!(v::Matrix{Float64}, u::AbstractVector, g::PICGrid{D}; m::Float64=1.0) where {D}
    N = size(v, 1)
    @threads :static for i in 1:N
        @inbounds p = SVector{D}(u[(i - 1) * 2D + D + d] for d in 1:D)
        vv = p2v(p; m=m)
        @inbounds for d in 1:D
            v[i, d] = vv[d]
        end
    end
    return v
end

"""
    make_periodic!(u, g, N)

Apply periodic boundaries to all particle positions in `u`.
"""
function make_periodic!(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    box = g.box
    L = extent(box)
    @threads :static for i in 1:N
        @inbounds for d in 1:D
            idx = (i - 1) * 2D + d
            u[idx] = mod(u[idx] - box.lo[d], L[d]) + box.lo[d]
        end
    end
    return u
end

"""
    coordinate_test(u, g, N)

Throw an error if any particle lies outside the box.
"""
function coordinate_test(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    box = g.box
    for d in 1:D
        vals = (u[(i-1)*2D + d] for i in 1:N)
        lo, hi = extrema(vals)
        if lo < box.lo[d] || hi > box.hi[d]
            error("particle out of box in dim $d: [$lo, $hi] not inside [$(box.lo[d]), $(box.hi[d])]")
        end
    end
    println("coordinate test passed")
end
