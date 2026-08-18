"""
Generic D-dimensional field interpolation at particle positions.
Computes the Lorentz force Fi = E - v × B using shape functions.
"""

function _grid_index_expr_interp(d::Int)
    off = Symbol(:off_, d)
    :(mod1(idx[i, $d] + $off, J[$d]))
end

function _weight_factor_expr_interp(d::Int, Order::Int, Factor::Int)
    off = Symbol(:off_, d)
    :(Shape(Val($Order), (-y[i, $d] + $off) / $Factor))
end

function _build_interp_body(D::Int, Order::Int, Factor::Int)
    grid_idx = [_grid_index_expr_interp(d) for d in 1:D]
    weight_expr = Expr(:call, :*, [_weight_factor_expr_interp(d, Order, Factor) for d in 1:D]...)
    norm = 1.0 / Factor^D

    # Build force vector expression depending on dimension.
    if D == 1
        ebv = Expr(:tuple, :(E[1, $(grid_idx[1])]))
    elseif D == 2
        ebv = Expr(:tuple,
            :(E[1, $(grid_idx[1]), $(grid_idx[2])] - v[i, 2] * B[$(grid_idx[1]), $(grid_idx[2])]),
            :(E[2, $(grid_idx[1]), $(grid_idx[2])] + v[i, 1] * B[$(grid_idx[1]), $(grid_idx[2])]))
    elseif D == 3
        ebv = Expr(:tuple,
            :(E[1, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]
              + v[i, 3] * B[2, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]
              - v[i, 2] * B[3, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]),
            :(E[2, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]
              + v[i, 1] * B[3, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]
              - v[i, 3] * B[1, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]),
            :(E[3, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]
              + v[i, 2] * B[1, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]
              - v[i, 1] * B[2, $(grid_idx[1]), $(grid_idx[2]), $(grid_idx[3])]))
    else
        error("interpolation only implemented for D = 1,2,3")
    end

    body = quote
        w = $weight_expr * $norm
        ebv = $ebv
        @inbounds for d in 1:$D
            Fi[i, d] += w * ebv[d]
        end
    end

    # Nest offset loops
    ex = body
    for d in D:-1:1
        off = Symbol(:off_, d)
        ex = :( for $off in $(-static_bound(Val(Order))*Factor):$(static_bound(Val(Order))*Factor + 1)
                    $ex
                end )
    end
    ex
end

@generated function _interpolate_kernel!(Fi::Matrix{Float64},
                                          E::AbstractArray{Float64},
                                          B::Union{Nothing,AbstractArray{Float64}},
                                          idx::Matrix{Int},
                                          y::Matrix{Float64},
                                          v::Matrix{Float64},
                                          J::SVector{D,Int},
                                          ::Val{Order},
                                          ::Val{Factor},
                                          istart::Int,
                                          istop::Int) where {D,Order,Factor}
    body = _build_interp_body(D, Order, Factor)
    quote
        for i in istart:istop
            $body
        end
        return nothing
    end
end

"""
    interpolate_field!(Fi, u, g; factor=1)

Interpolate the Lorentz force onto all particles and store in `Fi` (N×D).
"""
function interpolate_field!(Fi::Matrix{Float64}, u::AbstractVector, g::PICGrid{D};
                            factor::Int=1) where {D}
    N = particle_count(u, g)
    E = get_E(u, g, N)
    B = D ≥ 2 ? get_B(u, g, N) : nothing
    r = Matrix{Float64}(undef, N, D)
    idx = Matrix{Int}(undef, N, D)
    y = Matrix{Float64}(undef, N, D)
    v = Matrix{Float64}(undef, N, D)
    positions!(r, u, g)
    velocities!(v, u, g)
    get_indices_and_y!(idx, y, r, g)
    fill!(Fi, 0.0)
    nt = nthreads()
    @threads :static for tid in 1:nt
        istart, istop = _chunk_range(N, tid, nt)
        _interpolate_kernel!(Fi, E, B, idx, y, v, g.sz, Val(g.order), Val(factor), istart, istop)
    end
    return Fi
end

"""
    interpolate_field!(Fi, E, B, idx, y, v, g; factor=1)

Version that reuses precomputed indices/offsets/velocities.
"""
function interpolate_field!(Fi::Matrix{Float64},
                            E::AbstractArray{Float64},
                            B::Union{Nothing,AbstractArray{Float64}},
                            idx::Matrix{Int},
                            y::Matrix{Float64},
                            v::Matrix{Float64},
                            g::PICGrid{D}; factor::Int=1) where {D}
    N = size(Fi, 1)
    fill!(Fi, 0.0)
    nt = nthreads()
    @threads :static for tid in 1:nt
        istart, istop = _chunk_range(N, tid, nt)
        _interpolate_kernel!(Fi, E, B, idx, y, v, g.sz, Val(g.order), Val(factor), istart, istop)
    end
    return Fi
end
