"""
Energy and other diagnostic quantities.
"""

function kinetic_energy(u::AbstractVector, g::PICGrid{D}, N::Int; m::Float64=1.0) where {D}
    Ekin = 0.0
    @inbounds for i in 1:N
        p = SVector{D}(u[(i - 1) * 2D + D + d] for d in 1:D)
        Ekin += sqrt(m^2 + dot(p, p)) - m
    end
    return Ekin / N  # normalized per particle
end

function field_energy(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    F = reshape(view(u, particle_dof(g, N) + 1:length(u)), (field_components(g), size(g)...))
    E2 = 0.0
    B2 = 0.0
    vol = cell_volume(g)
    @inbounds for I in CartesianIndices(size(g))
        idx = I.I
        for d in 1:D
            E2 += F[d, idx...]^2
        end
        if D ≥ 2
            for b in (D + 1):field_components(g)
                B2 += F[b, idx...]^2
            end
        end
    end
    return 0.5 * vol * (E2 + B2)
end

function total_energy(u::AbstractVector, g::PICGrid{D}, N::Int; m::Float64=1.0) where {D}
    return kinetic_energy(u, g, N; m=m), field_energy(u, g, N)
end

function temperature_rel(u::AbstractVector, g::PICGrid{D}, N::Int; m::Float64=1.0) where {D}
    sv = zeros(D)
    sv2 = zeros(D)
    @inbounds for d in 1:D
        s = 0.0
        s2 = 0.0
        for i in 1:N
            pd = u[(i - 1) * 2D + D + d]
            s += pd
            s2 += pd^2
        end
        sv[d] = s
        sv2[d] = s2
    end
    return m * (sum(sv2) - dot(sv, sv) / N) / N / D
end

function total_momentum(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    P = zeros(D)
    @inbounds for d in 1:D
        s = 0.0
        for i in 1:N
            s += u[(i - 1) * 2D + D + d]
        end
        P[d] = s
    end
    return SVector{D}(P)
end

function total_charge(n::AbstractArray{Float64,D}, g::PICGrid{D}) where {D}
    return sum(n) * cell_volume(g)
end

"""
    temperature(u, g, N; mode=:rel, m=1.0)

Compute the kinetic temperature. `mode` can be:
- `:rel`   : relativistic definition `m (Σ <p_d^2> - <P>^2/N) / D`
- `:norel` : non-relativistic definition `Σ <p_d^2> / D`
"""
function temperature(u::AbstractVector, g::PICGrid{D}, N::Int; mode::Symbol=:rel, m::Float64=1.0) where {D}
    if mode == :rel
        return temperature_rel(u, g, N; m=m)
    elseif mode == :norel
        s2 = 0.0
        @inbounds for d in 1:D
            for i in 1:N
                pd = u[(i - 1) * 2D + D + d]
                s2 += pd^2
            end
        end
        return s2 / (N * D)
    else
        error("unknown temperature mode: $mode (use :rel or :norel)")
    end
end
