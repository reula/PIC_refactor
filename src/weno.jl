"""
WENO Maxwell solver (2D TM), ported from PIC-1D `choques_utils.jl`.

The fields are `(E1, E2, B)` and evolve as a conservation law
    ∂t U + ∂x Fx + ∂y Fy = 0
with the fluxes used in run_2D_clean.ipynb:
    Fx = (0,    B, E2)
    Fy = (-B, 0, -E1)

The spatial derivatives are approximated with a 5th-order WENO-Z scheme with
Lax–Friedrichs flux splitting (max wave speed = 1).  This mirrors
`WENOZreconstruction!`/`wenoz!` from PIC-1D, including the exact stencil
ordering and the signs of the reconstruction.
"""

struct WENOMaxwell{D} <: MaxwellSolver{D} end

# --- 3-component helper types -------------------------------------------------

@inline _fx(u::SVector{3,T}) where {T} = SVector(zero(T), u[3], u[2])   # (0, B, E2)
@inline _fy(u::SVector{3,T}) where {T} = SVector(-u[3], zero(T), -u[1]) # (-B, 0, -E1)

@inline function _field_at(F, i::Int, j::Int)
    SVector{3,Float64}(F[1, i, j], F[2, i, j], F[3, i, j])
end

"""
    _weno_z5_reconstruction(um2, um1, u0, up1, up2)

WENO-Z5 reconstruction of the left-biased interface value (cell averages
`um2..up2`, interface at `x_{j+1/2}`). Same formula as PIC-1D's
`WENOZreconstruction!`.
"""
@inline function _weno_z5_reconstruction(um2::SVector{3,T}, um1::SVector{3,T},
                                         u0::SVector{3,T}, up1::SVector{3,T},
                                         up2::SVector{3,T}) where {T}
    B1 = T(13) / T(12)
    B2 = T(1) / T(6)
    eps = T(1e-40)

    Q0 = T(2) * u0 + T(5) * up1 - up2
    Q1 = -um1 + T(5) * u0 + T(2) * up1
    Q2 = T(2) * um2 - T(7) * um1 + T(11) * u0

    s0 = u0 - T(2) * up1 + up2;        s1 = T(3) * u0 - T(4) * up1 + up2
    s2 = um1 - T(2) * u0 + up1;        s3 = um1 - up1
    s4 = um2 - T(2) * um1 + u0;        s5 = um2 - T(4) * um1 + T(3) * u0
    β0 = B1 * dot(s0, s0) + T(0.25) * dot(s1, s1)
    β1 = B1 * dot(s2, s2) + T(0.25) * dot(s3, s3)
    β2 = B1 * dot(s4, s4) + T(0.25) * dot(s5, s5)

    τ5 = abs(β2 - β0)

    α0 = T(0.3) * (one(T) + (τ5 / (β0 + eps))^2)
    α1 = T(0.6) * (one(T) + (τ5 / (β1 + eps))^2)
    α2 = T(0.1) * (one(T) + (τ5 / (β2 + eps))^2)
    asum = α0 + α1 + α2

    return (α0 * Q0 + α1 * Q1 + α2 * Q2) * B2 / asum
end

# -----------------------------------------------------------------------------
# Conservative WENO-Z5 + Lax-Friedrichs flux update along one direction
# -----------------------------------------------------------------------------

"""
    _weno_lf_sweep!(dF, F, g, axis)

Add the `-∂_axis F_axis` contribution of the Maxwell fluxes to `dF` using the
5th-order WENO-Z scheme with Lax–Friedrichs splitting (max speed 1), exactly
as PIC-1D's `wenoz!`.
"""
function _weno_lf_sweep!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                         g::PICGrid{2}, axis::Int)
    nx, ny = g.sz
    h = 1.0 / g.dx[axis]
    flux = axis == 1 ? _fx : _fy

    if axis == 1
        # x-sweep: for each row j, each i in 1:nx
        for j in 1:ny, i in 1:nx
            im3 = mod1(i - 3, nx); im2 = mod1(i - 2, nx); im1 = mod1(i - 1, nx)
            ip1 = mod1(i + 1, nx); ip2 = mod1(i + 2, nx); ip3 = mod1(i + 3, nx)

            u_m3 = _field_at(F, im3, j); u_m2 = _field_at(F, im2, j)
            u_m1 = _field_at(F, im1, j); u_0  = _field_at(F, i, j)
            u_p1 = _field_at(F, ip1, j); u_p2 = _field_at(F, ip2, j)
            u_p3 = _field_at(F, ip3, j)

            # Lax-Friedrichs splitting with S_MAX = 1:  F± = (flux ± U)/2
            FP_m3 = (flux(u_m3) + u_m3) / 2; FM_m3 = (flux(u_m3) - u_m3) / 2
            FP_m2 = (flux(u_m2) + u_m2) / 2; FM_m2 = (flux(u_m2) - u_m2) / 2
            FP_m1 = (flux(u_m1) + u_m1) / 2; FM_m1 = (flux(u_m1) - u_m1) / 2
            FP_0  = (flux(u_0)  + u_0)  / 2; FM_0  = (flux(u_0)  - u_0)  / 2
            FP_p1 = (flux(u_p1) + u_p1) / 2; FM_p1 = (flux(u_p1) - u_p1) / 2
            FP_p2 = (flux(u_p2) + u_p2) / 2; FM_p2 = (flux(u_p2) - u_p2) / 2
            FP_p3 = (flux(u_p3) + u_p3) / 2; FM_p3 = (flux(u_p3) - u_p3) / 2

            # interfaces i-1/2 and i+1/2
            H_m = _weno_z5_reconstruction(FP_m3, FP_m2, FP_m1, FP_0, FP_p1) +
                  _weno_z5_reconstruction(FM_p2, FM_p1, FM_0, FM_m1, FM_m2)
            H_p = _weno_z5_reconstruction(FP_m2, FP_m1, FP_0, FP_p1, FP_p2) +
                  _weno_z5_reconstruction(FM_p3, FM_p2, FM_p1, FM_0, FM_m1)

            dU = -h * (H_p - H_m)
            dF[1, i, j] += dU[1]
            dF[2, i, j] += dU[2]
            dF[3, i, j] += dU[3]
        end
    else
        # y-sweep
        for j in 1:ny, i in 1:nx
            jm3 = mod1(j - 3, ny); jm2 = mod1(j - 2, ny); jm1 = mod1(j - 1, ny)
            jp1 = mod1(j + 1, ny); jp2 = mod1(j + 2, ny); jp3 = mod1(j + 3, ny)

            u_m3 = _field_at(F, i, jm3); u_m2 = _field_at(F, i, jm2)
            u_m1 = _field_at(F, i, jm1); u_0  = _field_at(F, i, j)
            u_p1 = _field_at(F, i, jp1); u_p2 = _field_at(F, i, jp2)
            u_p3 = _field_at(F, i, jp3)

            FP_m3 = (flux(u_m3) + u_m3) / 2; FM_m3 = (flux(u_m3) - u_m3) / 2
            FP_m2 = (flux(u_m2) + u_m2) / 2; FM_m2 = (flux(u_m2) - u_m2) / 2
            FP_m1 = (flux(u_m1) + u_m1) / 2; FM_m1 = (flux(u_m1) - u_m1) / 2
            FP_0  = (flux(u_0)  + u_0)  / 2; FM_0  = (flux(u_0)  - u_0)  / 2
            FP_p1 = (flux(u_p1) + u_p1) / 2; FM_p1 = (flux(u_p1) - u_p1) / 2
            FP_p2 = (flux(u_p2) + u_p2) / 2; FM_p2 = (flux(u_p2) - u_p2) / 2
            FP_p3 = (flux(u_p3) + u_p3) / 2; FM_p3 = (flux(u_p3) - u_p3) / 2

            H_m = _weno_z5_reconstruction(FP_m3, FP_m2, FP_m1, FP_0, FP_p1) +
                  _weno_z5_reconstruction(FM_p2, FM_p1, FM_0, FM_m1, FM_m2)
            H_p = _weno_z5_reconstruction(FP_m2, FP_m1, FP_0, FP_p1, FP_p2) +
                  _weno_z5_reconstruction(FM_p3, FM_p2, FM_p1, FM_0, FM_m1)

            dU = -h * (H_p - H_m)
            dF[1, i, j] += dU[1]
            dF[2, i, j] += dU[2]
            dF[3, i, j] += dU[3]
        end
    end
    return dF
end

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{1}, m::WENOMaxwell{1})
    # 1D electrostatic: Maxwell waves not evolved
    return dF
end

function add_maxwell_rhs!(dF::AbstractArray{Float64}, F::AbstractArray{Float64},
                          g::PICGrid{2}, m::WENOMaxwell{2})
    _weno_lf_sweep!(dF, F, g, 1)
    _weno_lf_sweep!(dF, F, g, 2)
    return dF
end
