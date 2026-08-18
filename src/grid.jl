"""Volume of a Box."""
volume(b::Box) = prod(extent(b))
volume(g::PICGrid) = cell_volume(g) * n_cells(g)

"""Uniform cell sizes. For periodic domains N cells span [lo, hi]."""
function differentials(b::Box, sz::AbstractVector{<:Integer})
    D = length(b)
    SVector{D}(extent(b) ./ sz)
end

differentials(g::PICGrid) = g.dx

"""
    get_index_and_y(s, J, L)

For a coordinate `s` in a periodic domain of length `L` divided into `J` cells,
return (j, y) where j is the 1-based cell index and y ∈ [0,1) is the offset
from the left grid point. `yshift` subtracts a constant from y (used for staggered
midpoint deposition).
"""
@inline function get_index_and_y(s::Real, J::Integer, L::Real; yshift::Real=0.0)
    tmp = (s / L * J + J) % J
    j = floor(Int64, tmp) + 1
    y = (tmp % 1) - yshift
    return j, y
end

@inline function get_index_and_y!(j::AbstractVector{Int}, y::AbstractVector{T}, r, sz, L; yshift::Real=0.0) where {T}
    for d in eachindex(j)
        j[d], y[d] = get_index_and_y(r[d], sz[d], L[d]; yshift=yshift)
    end
    return j, y
end

"""
    get_indices_and_y!(idx, y, r, g; yshift=0.0)

Vectorized version. `r` is N×D matrix of particle positions.
`idx` and `y` are N×D output arrays.
"""
function get_indices_and_y!(idx::Matrix{Int}, y::Matrix{Float64}, r::Matrix{Float64}, g::PICGrid{D}; yshift::Real=0.0) where {D}
    N = size(r, 1)
    L = extent(g.box)
    sz = g.sz
    @threads :static for i in 1:N
        @inbounds for d in 1:D
            tmp = (r[i, d] / L[d] * sz[d] + sz[d]) % sz[d]
            idx[i, d] = floor(Int64, tmp) + 1
            y[i, d] = (tmp % 1) - yshift
        end
    end
    return idx, y
end

"""
    wrap_index(j, sz)

Apply periodic wrapping per dimension for 1-based index `j`.
"""
@inline wrap_index(j, sz) = mod1(j, sz)
