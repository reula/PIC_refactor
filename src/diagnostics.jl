"""
Energy and other diagnostic quantities.
"""

function kinetic_energy(u::AbstractVector, g::PICGrid{D}, N::Int; m::Float64=1.0) where {D}
    vol = volume(g)
    Ekin = 0.0
    @inbounds for i in 1:N
        p = SVector{D}(u[(i - 1) * 2D + D + d] for d in 1:D)
        Ekin += sqrt(m^2 + dot(p, p)) - m
    end
    return Ekin / N * vol  # normalized per particle
end

function field_energy(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    F = reshape(view(u, particle_dof(g, N) + 1:length(u)), (field_components(g), size(g)...))
    E2 = 0.0
    B2 = 0.0
    cell_vol = cell_volume(g)
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
    return 0.5 * cell_vol * (E2 + B2)
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
    check_constraints(u, g, N, dep, factor)

Check Gauss's law (div E = -rho, i.e. div E + rho = 0) for the state `u`.
Returns `(err_max, err_l2)` using the cell volume for the L2 norm.
In 2D TM mode the magnetic field is scalar and divergence-free trivially.
"""
function check_constraints(u::AbstractVector, g::PICGrid{D}, N::Int, dep::DepositDensity{D}, factor::Int) where {D}
    E, B = get_fields(u, g, N)
    n = zeros(size(g))
    deposit!(dep, u, g, n; factor=factor)
    rho = n .- 1.0
    divE = field_divergence(E, g)
    residual = divE .+ rho
    ave = abs.(divE) .+ abs.(rho)
    err_max = maximum(abs.(residual))/maximum(ave)
    err_l2 = sqrt(sum(residual.^2)/sum(ave.^2))
    return err_max, err_l2
end

# ---------------------------------------------------------------------------
# Weak-form Gauss-law constraint diagnostics
# ---------------------------------------------------------------------------

function _phi_test(x::AbstractVector, x0::AbstractVector, box::Box{D}, r0::Real, p::Real) where {D}
    L = extent(box)
    desp = zeros(D)
    dx = zeros(D)
    @assert 0.0 < abs(r0) && abs(r0) < maximum(L)
    for d in 1:D
        lo = box.lo[d]
        hi = box.hi[d]
        if x0[d] - r0 < lo
            desp[d] = r0
        elseif x0[d] + r0 > hi
            desp[d] = -r0
        else
            desp[d] = 0.0
        end
        dx[d] = (x[d] - x0[d] + desp[d]) % L[d] - desp[d]
    end
    r2 = dot(dx, dx)
    r02 = r0^2
    if r2 < r02
        return (r2 - r02)^p / r02^(p + 1) / π * (p + 1)
    else
        return 0.0
    end
end

function _grad_phi_test(x::AbstractVector, x0::AbstractVector, box::Box{D}, r0::Real, p::Real) where {D}
    L = extent(box)
    desp = zeros(D)
    dx = zeros(D)
    @assert 0.0 < abs(r0) && abs(r0) < maximum(L)
    for d in 1:D
        lo = box.lo[d]
        hi = box.hi[d]
        if x0[d] - r0 < lo
            desp[d] = r0
        elseif x0[d] + r0 > hi
            desp[d] = -r0
        else
            desp[d] = 0.0
        end
        dx[d] = (x[d] - x0[d] + desp[d]) % L[d] - desp[d]
    end
    r2 = dot(dx, dx)
    r02 = r0^2
    if r2 < r02
        return 2p * (r2 - r02)^(p - 1) / r02^(p + 1) * dx / π * (p + 1)
    else
        return zeros(D)
    end
end

"""
    check_constraints_weak(u, g, N, dep, factor, x0, r0_frac, p)

Weak-form check of Gauss's law: ∫ E·∇φ dV = ∫ ρ φ dV for a compact test
function φ centred at `x0`. The support radius is `r0_frac * minimum(extent(g.box))`.
Returns `(divE_term, rho_term, err_norm, err_abs)` matching PIC-1D's
`constraint_test` convention.
"""
function check_constraints_weak(u::AbstractVector, g::PICGrid{D}, N::Int, dep::DepositDensity{D},
                                factor::Real, x0::AbstractVector, r0_frac::Real, p::Real) where {D}
    E, _ = get_fields(u, g, N)
    n = zeros(size(g))
    deposit!(dep, u, g, n; factor=factor)
    rho = n .- 1.0

    r0 = r0_frac * minimum(extent(g.box))
    lo = g.box.lo
    dx = g.dx

    divE_term = 0.0
    rho_term = 0.0
    norm_divE = 0.0
    norm_rho = 0.0

    for I in CartesianIndices(size(g))
        idx = I.I
        x = SVector{D}(lo[d] + (idx[d] - 1) * dx[d] for d in 1:D)
        x0_s = SVector{D}(x0[d] for d in 1:D)
        phi = _phi_test(x, x0_s, g.box, r0, p)
        grad_phi = _grad_phi_test(x, x0_s, g.box, r0, p)
        E_cell = SVector{D}(E[d, idx...] for d in 1:D)
        Edotgrad = dot(E_cell, grad_phi)
        rho_phi = rho[idx...] * phi
        divE_term += Edotgrad
        rho_term += rho_phi
        norm_divE += abs(Edotgrad)
        norm_rho += abs(rho_phi)
    end

    residual = divE_term - rho_term
    denom = norm_divE + norm_rho
    err_norm = denom > 0 ? abs(residual) / denom : 0.0
    err_abs = abs(residual) * prod(dx)
    return divE_term, rho_term, err_norm, err_abs
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
