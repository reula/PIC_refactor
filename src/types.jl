"""
    Box{D}

Rectangular periodic domain in D dimensions.
`lo[d]` and `hi[d]` are the lower/upper bounds along dimension d.
"""
struct Box{D}
    lo::SVector{D,Float64}
    hi::SVector{D,Float64}
    function Box{D}(lo::AbstractVector, hi::AbstractVector) where {D}
        @assert D ≥ 1
        new{D}(SVector{D}(lo), SVector{D}(hi))
    end
end

Box(lo::Tuple, hi::Tuple) = Box{length(lo)}(collect(lo), collect(hi))

function Box(bounds::NTuple{L,Real}) where {L}
    iseven(L) || error("Box expects an even number of bounds")
    D = L ÷ 2
    return Box{D}(collect(bounds[1:2:end]), collect(bounds[2:2:end]))
end


Base.length(::Box{D}) where {D} = D
Base.ndims(::Box{D}) where {D} = D
extent(b::Box) = b.hi .- b.lo
extent(b::Box, d::Int) = b.hi[d] - b.lo[d]

"""
    PICGrid{D}

Uniform periodic grid with `size[J]` cells per dimension and `order` shape order.
`dx[d]` is the uniform cell size (periodic, so N cells from lo to hi).
"""
struct PICGrid{D}
    box::Box{D}
    sz::SVector{D,Int}
    dx::SVector{D,Float64}
    order::Int
    function PICGrid{D}(box::Box{D}, sz::AbstractVector{<:Integer}, order::Integer) where {D}
        @assert all(>(0), sz)
        @assert 1 ≤ order ≤ 6
        dx = extent(box) ./ sz
        new{D}(box, SVector{D}(Int.(sz)), SVector{D}(dx), Int(order))
    end
end

PICGrid(box::Box{D}, sz, order) where {D} = PICGrid{D}(box, collect(sz), order)

Base.ndims(::PICGrid{D}) where {D} = D
Base.size(g::PICGrid) = Tuple(g.sz)
Base.size(g::PICGrid, d::Int) = g.sz[d]
n_cells(g::PICGrid) = prod(g.sz)
cell_volume(g::PICGrid) = prod(g.dx)

n_B_components(::PICGrid{1}) = 0
n_B_components(::PICGrid{2}) = 1
n_B_components(::PICGrid{3}) = 3

field_components(g::PICGrid) = ndims(g) + n_B_components(g)
state_length(g::PICGrid, N::Int) = 2 * ndims(g) * N + field_components(g) * n_cells(g)
particle_dof(g::PICGrid, N::Int) = 2 * ndims(g) * N
field_dof(g::PICGrid) = field_components(g) * n_cells(g)

@inline function range_x(i::Int, ::PICGrid{D}) where {D}
    base = (i - 1) * 2D
    (base + 1):(base + D)
end

@inline function range_p(i::Int, ::PICGrid{D}) where {D}
    base = (i - 1) * 2D
    (base + D + 1):(base + 2D)
end

"""
    range_F(g, N)

Linear range covering the whole field block in the state vector. The block is
stored as a single array reshaped to `(field_components(g), size(g)...)`, i.e.
component-interleaved per cell. This matches the layout used in
PIC-1D/run_2D_clean.ipynb.
"""
@inline range_F(g::PICGrid, N::Int) = (particle_dof(g, N) + 1):state_length(g, N)

@inline function _reshape_fields(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    reshape(view(u, range_F(g, N)), (field_components(g), size(g)...))
end

function get_E(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    F = _reshape_fields(u, g, N)
    idx = ntuple(_ -> :, D)
    return view(F, 1:D, idx...)
end

function get_B(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    D == 1 && error("No magnetic field in 1D")
    F = _reshape_fields(u, g, N)
    idx = ntuple(_ -> :, D)
    if D == 2
        return view(F, D + 1, idx...)
    else
        return view(F, (D + 1):(2D), idx...)
    end
end

make_state(g::PICGrid, N::Int) = zeros(Float64, state_length(g, N))

function set_fields!(u::AbstractVector, g::PICGrid{D}, N::Int, E::AbstractArray, B=nothing) where {D}
    F = _reshape_fields(u, g, N)
    idx = ntuple(_ -> :, D)
    view(F, 1:D, idx...) .= E
    if D ≥ 2
        B isa Nothing && error("B field required for D=$D")
        if D == 2
            view(F, D + 1, idx...) .= B
        else
            view(F, (D + 1):(2D), idx...) .= B
        end
    end
    u
end

function get_fields(u::AbstractVector, g::PICGrid{D}, N::Int) where {D}
    E = get_E(u, g, N)
    B = D ≥ 2 ? get_B(u, g, N) : nothing
    return E, B
end

struct SimParams{Grid<:PICGrid, DepoD<:Any, DepoC<:Any}
    grid::Grid
    N::Int
    dt::Float64
    use_maxwell::Bool
    deposit_density!::DepoD
    deposit_current!::DepoC
    factor::Float64
end
