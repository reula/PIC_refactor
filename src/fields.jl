"""
Field solvers and Maxwell RHS.
"""

# -----------------------------------------------------------------------------
# Poisson solver (periodic, zero-mean)
# -----------------------------------------------------------------------------

function _k_vectors(g::PICGrid{D}) where {D}
    k1 = collect(rfftfreq(g.sz[1], 2π / g.dx[1]))
    if D == 1
        return (k1,)
    end
    kother = ntuple(d -> collect(fftfreq(g.sz[d + 1], 2π / g.dx[d + 1])), D - 1)
    return (k1, kother...)
end

function _k2_array(g::PICGrid{D}) where {D}
    ks = _k_vectors(g)
    shape = ntuple(d -> length(ks[d]), D)
    k2 = Array{Float64}(undef, shape)
    for I in CartesianIndices(k2)
        s = 0.0
        @inbounds for d in 1:D
            s += ks[d][I[d]]^2
        end
        k2[I] = s
    end
    k2[1] = 1.0  # avoid division by zero; mode is set to zero explicitly
    return k2
end

"""
    solve_poisson!(E, rho, g)

Solve ∇·E = -rho on a periodic grid and store the vector field E.
`E` must have shape `(D, size(g)...)`. The zero mode is removed.

Note: with the convention rho = n - 1 used in PIC-1D, this is the standard
Poisson equation for electrons (charge density = -rho).
"""
function solve_poisson!(E::AbstractArray{Float64,Dp}, rho::AbstractArray{Float64,D},
                        g::PICGrid{D}) where {D,Dp}
    @assert Dp == D + 1
    @assert size(E)[2:end] == Tuple(g.sz)
    V = rfft(rho)
    ks = _k_vectors(g)
    k2 = _k2_array(g)
    for d in 1:D
        Vd = similar(V)
        for I in CartesianIndices(V)
            kd = ks[d][I[d]]
            Vd[I] = im * kd * V[I] / k2[I]
        end
        Vd[1] = 0.0 + 0.0im
        Ed = irfft(Vd, g.sz[1])
        E_slice = selectdim(E, 1, d)
        E_slice .= Ed
    end
    return E
end

# -----------------------------------------------------------------------------
# Maxwell solvers
# -----------------------------------------------------------------------------

abstract type MaxwellSolver{D} end

struct NoMaxwell{D} <: MaxwellSolver{D} end

# -----------------------------------------------------------------------------
# Spectral Maxwell solver (periodic uniform grid)
# -----------------------------------------------------------------------------

struct SpectralMaxwell{D} <: MaxwellSolver{D}
    ks::NTuple{D,Vector{Float64}}
    plan_rfft::Any
    plan_irfft::Any
end

function SpectralMaxwell(g::PICGrid{D}) where {D}
    ks = _k_vectors(g)
    A = zeros(Float64, Tuple(g.sz))
    plan_rfft = FFTW.plan_rfft(A)
    plan_irfft = FFTW.plan_irfft(plan_rfft * A, g.sz[1])
    return SpectralMaxwell{D}(ks, plan_rfft, plan_irfft)
end

function _scale_k!(Ahat::AbstractArray{ComplexF64,D}, k::Vector{Float64}, dir::Int) where {D}
    # Multiply Ahat by im*k along direction dir.
    shape = ntuple(d -> d == dir ? length(k) : size(Ahat, d), D)
    # use broadcasting with reshaped k
    k_reshaped = reshape(k, ntuple(d -> d == dir ? length(k) : 1, D))
    @inbounds Ahat .*= (im .* k_reshaped)
    return Ahat
end

function _spectral_derivative(A::AbstractArray{Float64}, m::SpectralMaxwell, dir::Int)
    Ac = copy(A)  # FFTW plans need contiguous memory
    Ahat = m.plan_rfft * Ac
    _scale_k!(Ahat, m.ks[dir], dir)
    return m.plan_irfft * Ahat
end

"""
    field_divergence(E, g; mode=:spectral)

Compute the divergence of a vector field `E` with shape `(D, size(g)...)`
using spectral derivatives on the periodic grid `g`.
Returns an array with shape `size(g)`.
"""
function field_divergence(E::AbstractArray{Float64}, g::PICGrid{D}; mode::Symbol=:spectral) where {D}
    if mode != :spectral
        error("field_divergence only supports mode=:spectral")
    end
    spec = SpectralMaxwell(g)
    divE = zeros(size(g))
    for d in 1:D
        Ed = selectdim(E, 1, d)
        divE .+= _spectral_derivative(Ed, spec, d)
    end
    return divE
end

function _add_derivative!(out, A, m::SpectralMaxwell{D}, dir::Int, α::Real=1.0) where {D}
    out .+= α .* _spectral_derivative(A, m, dir)
end

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{2}, m::SpectralMaxwell{2})
    # F layout: [E1, E2, B3]
    @views begin
        E1 = F[1, :, :]
        E2 = F[2, :, :]
        B  = F[3, :, :]
        dE1 = dF[1, :, :]
        dE2 = dF[2, :, :]
        dB  = dF[3, :, :]
    end
    # ∂t E = ∇×B - J   (J added elsewhere)
    _add_derivative!(dE1, B, m, 2, 1.0)
    _add_derivative!(dE2, B, m, 1, -1.0)
    # ∂t B = -∇×E
    _add_derivative!(dB, E1, m, 2, 1.0)
    _add_derivative!(dB, E2, m, 1, -1.0)
    return dF
end

const _lc = zeros(Int, 3, 3, 3)
_lc[1, 2, 3] = 1; _lc[1, 3, 2] = -1
_lc[2, 3, 1] = 1; _lc[2, 1, 3] = -1
_lc[3, 1, 2] = 1; _lc[3, 2, 1] = -1

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{3}, m::SpectralMaxwell{3})
    # F layout: [E1,E2,E3,B1,B2,B3]
    D = 3
    @views for d in 1:D
        Ed = F[d, :, :, :]
        Bd = F[D + d, :, :, :]
        dEd = dF[d, :, :, :]
        dBd = dF[D + d, :, :, :]
        for e in 1:D
            for f in 1:D
                c = _lc[d, e, f]
                if c != 0
                    _add_derivative!(dEd, Bd, m, e, c)
                    _add_derivative!(dBd, Ed, m, e, -c)
                end
            end
        end
    end
    return dF
end

function add_maxwell_rhs!(dF, F, g::PICGrid{1}, ::NoMaxwell{1})
    # No magnetic field in 1D
    return dF
end

function add_maxwell_rhs!(dF, F, g::PICGrid{1}, ::SpectralMaxwell{1})
    # No magnetic field in 1D
    return dF
end

# -----------------------------------------------------------------------------
# Derivative-operator Maxwell solvers
# (SBP, Holoborodko and Lanczos use one periodic derivative operator per axis;
# they only differ in which operator type was requested at construction time.)
# -----------------------------------------------------------------------------

# Shared 2D TM Maxwell RHS for any solver that provides (Dx, Dy):
#   ∂t E1 =  ∂y B         ∂t E2 = -∂x B        ∂t B =  ∂y E1 - ∂x E2
function _derivative_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                                  g::PICGrid{2}, Dx, Dy)
    @views for i in 1:g.sz[1]
        mul!(dF[1, i, :], Dy, F[3, i, :], 1.0, 1.0)   # +∂y B
        mul!(dF[3, i, :], Dy, F[1, i, :], 1.0, 1.0)   # +∂y E1
    end
    @views for j in 1:g.sz[2]
        mul!(dF[2, :, j], Dx, F[3, :, j], -1.0, 1.0)  # -∂x B
        mul!(dF[3, :, j], Dx, F[2, :, j], -1.0, 1.0)  # -∂x E2
    end
    return dF
end

struct SBPMaxwell{D} <: MaxwellSolver{D}
    Dx::NTuple{D,Any}           # derivative operators
    Δx::NTuple{D,Any}           # optional dissipation operators (or zeros)
    σ::NTuple{D,Float64}        # dissipation strengths
    use_dissipation::Bool
end

function SBPMaxwell(derivative_ops; dissipation_ops=nothing, σ=nothing)
    D = length(derivative_ops)
    Δx = dissipation_ops === nothing ? ntuple(d -> 0.0, D) : Tuple(dissipation_ops)
    σs = σ === nothing ? ntuple(d -> 0.0, D) : Tuple(σ)
    return SBPMaxwell{D}(Tuple(derivative_ops), Δx, σs, dissipation_ops !== nothing)
end

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{2}, m::SBPMaxwell{2})
    Dx, Dy = m.Dx
    _derivative_maxwell_rhs!(dF, F, g, Dx, Dy)
    if m.use_dissipation
        Δx, Δy = m.Δx
        σx, σy = m.σ
        @views for i in 1:g.sz[1]
            for l in 1:3
                mul!(dF[l, i, :], Δy, F[l, i, :], σy, 1.0)
            end
        end
        @views for j in 1:g.sz[2]
            for l in 1:3
                mul!(dF[l, :, j], Δx, F[l, :, j], σx, 1.0)
            end
        end
    end
    return dF
end

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{1}, m::SBPMaxwell{1})
    # 1D electrostatic: Maxwell waves not evolved
    return dF
end

"""
    HolboMaxwell(Dops)

Maxwell solver based on the Holoborodko periodic derivative operators
(one per spatial axis), see `periodic_derivative_operator(Holoborodko2008(); …)`
from SummationByPartsOperators.
"""
struct HolboMaxwell{D} <: MaxwellSolver{D}
    Dx::NTuple{D,Any}
end
HolboMaxwell(derivative_ops) = HolboMaxwell{length(derivative_ops)}(Tuple(derivative_ops))

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{2}, m::HolboMaxwell{2})
    Dx, Dy = m.Dx
    return _derivative_maxwell_rhs!(dF, F, g, Dx, Dy)
end

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{1}, m::HolboMaxwell{1})
    # 1D electrostatic: Maxwell waves not evolved
    return dF
end

"""
    LanczosMaxwell(Dops)

Maxwell solver based on the Lanczos (low-noise) periodic derivative operators
(one per spatial axis), see `periodic_derivative_operator(LanczosLowNoise(); …)`
from SummationByPartsOperators.
"""
struct LanczosMaxwell{D} <: MaxwellSolver{D}
    Dx::NTuple{D,Any}
end
LanczosMaxwell(derivative_ops) = LanczosMaxwell{length(derivative_ops)}(Tuple(derivative_ops))

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{2}, m::LanczosMaxwell{2})
    Dx, Dy = m.Dx
    return _derivative_maxwell_rhs!(dF, F, g, Dx, Dy)
end

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{1}, m::LanczosMaxwell{1})
    # 1D electrostatic: Maxwell waves not evolved
    return dF
end
