"""
Generic D-dimensional particle deposition (density and current) with threading.
Specialized via generated functions for D = 1, 2, 3 and compile-time order/factor.
Threading is handled outside the generated kernels.
"""

# -----------------------------------------------------------------------------
# Helpers to build generated loop nests
# -----------------------------------------------------------------------------

function _grid_index_expr(d::Int)
    off = Symbol(:off_, d)
    :(mod1(idx[i, $d] + $off, J[$d]))
end

function _weight_factor_expr(d::Int, Order::Int, Factor::Int)
    off = Symbol(:off_, d)
    :(Shape(Val($Order), (-y[i, $d] + $off) / $Factor))
end

function _build_density_body(D::Int, Order::Int, Factor::Int)
    grid_idx = Expr(:tuple, (_grid_index_expr(d) for d in 1:D)...)
    weight_expr = Expr(:call, :*, (_weight_factor_expr(d, Order, Factor) for d in 1:D)...)
    :( @inbounds local_results[$grid_idx..., tid] += $weight_expr / n0 )
end

function _build_density_loop(D::Int, h::Int, body::Expr)
    ex = body
    for d in D:-1:1
        off = Symbol(:off_, d)
        ex = :( for $off in $(-h):$(h+1)
                    $ex
                end )
    end
    ex
end

function _build_current_body(D::Int, Order::Int, Factor::Int)
    grid_idx = Expr(:tuple, (_grid_index_expr(d) for d in 1:D)...)
    weight_expr = Expr(:call, :*, (_weight_factor_expr(d, Order, Factor) for d in 1:D)...)
    :( @inbounds for d in 1:$D
           local_results[$grid_idx..., d, tid] += $weight_expr * v[i, d] / n0
       end )
end

function _build_current_loop(D::Int, h::Int, body::Expr)
    ex = body
    for d in D:-1:1
        off = Symbol(:off_, d)
        ex = :( for $off in $(-h):$(h+1)
                    $ex
                end )
    end
    ex
end

# -----------------------------------------------------------------------------
# Chunking helpers
# -----------------------------------------------------------------------------

function _chunk_range(N::Int, tid::Int, nt::Int)
    chunk = div(N, nt)
    rem = N - chunk * nt
    if tid <= rem
        start = (tid - 1) * (chunk + 1) + 1
        stop  = start + chunk
    else
        start = rem * (chunk + 1) + (tid - rem - 1) * chunk + 1
        stop  = start + chunk - 1
    end
    return start, stop
end

# -----------------------------------------------------------------------------
# Density deposition kernel (generated, no threading inside)
# -----------------------------------------------------------------------------

@generated function _deposit_density_kernel!(local_results::Array{Float64},
                                              idx::Matrix{Int},
                                              y::Matrix{Float64},
                                              J::SVector{D,Int},
                                              ::Val{Order},
                                              ::Val{Factor},
                                              n0::Float64,
                                              istart::Int,
                                              istop::Int,
                                              tid::Int) where {D,Order,Factor}
    h = static_bound(Val(Order)) * Factor
    body = _build_density_body(D, Order, Factor)
    loop = _build_density_loop(D, h, body)
    quote
        for i in istart:istop
            $loop
        end
        return nothing
    end
end

# -----------------------------------------------------------------------------
# Current deposition kernel (generated, no threading inside)
# -----------------------------------------------------------------------------

@generated function _deposit_current_kernel!(local_results::Array{Float64},
                                              idx::Matrix{Int},
                                              y::Matrix{Float64},
                                              v::Matrix{Float64},
                                              J::SVector{D,Int},
                                              ::Val{Order},
                                              ::Val{Factor},
                                              n0::Float64,
                                              istart::Int,
                                              istop::Int,
                                              tid::Int) where {D,Order,Factor}
    h = static_bound(Val(Order)) * Factor
    body = _build_current_body(D, Order, Factor)
    loop = _build_current_loop(D, h, body)
    quote
        for i in istart:istop
            $loop
        end
        return nothing
    end
end

# -----------------------------------------------------------------------------
# DepositDensity struct
# -----------------------------------------------------------------------------

mutable struct DepositDensity{D}
    N::Int
    J::SVector{D,Int}
    r::Matrix{Float64}
    idx::Matrix{Int}
    y::Matrix{Float64}
    local_results::Array{Float64}
    function DepositDensity(g::PICGrid{D}, N::Int) where {D}
        J = g.sz
        new{D}(N, J, zeros(Float64, N, D), ones(Int, N, D), zeros(Float64, N, D),
               zeros(Float64, J..., nthreads()))
    end
end

function reset!(dep::DepositDensity)
    fill!(dep.local_results, 0.0)
    return dep
end

function _reduce_density!(n::AbstractArray{Float64,D}, local_results::Array{Float64}) where {D}
    fill!(n, 0.0)
    nt = size(local_results, D + 1)
    @inbounds for tid in 1:nt
        for I in CartesianIndices(size(n))
            n[I] += local_results[I.I..., tid]
        end
    end
    return n
end

"""
    deposit!(dep::DepositDensity, u, g, n; yshift=0.0, factor=1)

Deposit particle density into `n` (array with shape `size(g)`).
"""
function deposit!(dep::DepositDensity{D}, u::AbstractVector, g::PICGrid{D},
                  n::AbstractArray{Float64,D}; yshift::Real=0.0, factor::Int=1) where {D}
    @assert dep.N == particle_count(u, g)
    positions!(dep.r, u, g)
    get_indices_and_y!(dep.idx, dep.y, dep.r, g; yshift=yshift)
    reset!(dep)
    n0 = dep.N / n_cells(g) * factor^2
    nt = nthreads()
    @threads :static for tid in 1:nt
        istart, istop = _chunk_range(dep.N, tid, nt)
        _deposit_density_kernel!(dep.local_results, dep.idx, dep.y, g.sz,
                                  Val(g.order), Val(factor), n0, istart, istop, tid)
    end
    _reduce_density!(n, dep.local_results)
    return n
end

# -----------------------------------------------------------------------------
# DepositCurrent struct
# -----------------------------------------------------------------------------

mutable struct DepositCurrent{D}
    N::Int
    J::SVector{D,Int}
    r::Matrix{Float64}
    v::Matrix{Float64}
    idx::Matrix{Int}
    y::Matrix{Float64}
    local_results::Array{Float64}
    function DepositCurrent(g::PICGrid{D}, N::Int) where {D}
        J = g.sz
        new{D}(N, J, zeros(Float64, N, D), zeros(Float64, N, D), ones(Int, N, D),
               zeros(Float64, N, D), zeros(Float64, J..., D, nthreads()))
    end
end

function reset!(dep::DepositCurrent)
    fill!(dep.local_results, 0.0)
    return dep
end

function _reduce_current!(S::AbstractArray{Float64}, local_results::Array{Float64})
    D = size(S, 1)
    sz_grid = size(S)[2:end]
    fill!(S, 0.0)
    nt = size(local_results, D + 2)
    @inbounds for tid in 1:nt
        for Igrid in CartesianIndices(sz_grid)
            idx_grid = Igrid.I
            for d in 1:D
                S[d, idx_grid...] += local_results[idx_grid..., d, tid]
            end
        end
    end
    return S
end

"""
    deposit!(dep::DepositCurrent, u, g, S; yshift=0.0, factor=1, m=1.0)

Deposit particle current into `S` (array with shape `(D, size(g)...)`).
"""
function deposit!(dep::DepositCurrent{D}, u::AbstractVector, g::PICGrid{D},
                  S::AbstractArray{Float64,Dp}; yshift::Real=0.0,
                  factor::Int=1, m::Float64=1.0) where {D,Dp}
    @assert dep.N == particle_count(u, g)
    @assert Dp == D + 1
    positions!(dep.r, u, g)
    velocities!(dep.v, u, g; m=m)
    get_indices_and_y!(dep.idx, dep.y, dep.r, g; yshift=yshift)
    reset!(dep)
    n0 = dep.N / n_cells(g) * factor^2
    nt = nthreads()
    @threads :static for tid in 1:nt
        istart, istop = _chunk_range(dep.N, tid, nt)
        _deposit_current_kernel!(dep.local_results, dep.idx, dep.y, dep.v, g.sz,
                                  Val(g.order), Val(factor), n0, istart, istop, tid)
    end
    _reduce_current!(S, dep.local_results)
    return S
end

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

@inline function particle_count(u::AbstractVector, g::PICGrid{D}) where {D}
    div(length(u) - field_dof(g), 2D)
end
