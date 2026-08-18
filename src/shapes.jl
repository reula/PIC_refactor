"""
Shape / weight functions.
`W` is the B-spline-like weight of order 0..5 (support grows with order).
`Shape` is `W` of one order lower, used for particle deposition.
"""

@inline function W_shape(::Val{0}, y::T)::T where {T<:AbstractFloat}
    y = abs(y)
    y <= T(0.5) ? T(1.0) : T(0.0)
end

@inline function W_shape(::Val{1}, y::T)::T where {T<:AbstractFloat}
    y = abs(y)
    y <= T(1.0) ? T(1.0) - y : T(0.0)
end

@inline function W_shape(::Val{2}, y::T)::T where {T<:AbstractFloat}
    y = abs(y)
    if y <= T(0.5)
        T(0.75) - y * y
    elseif y <= T(1.5)
        (T(3.0) - T(2.0) * y)^2 / T(8.0)
    else
        T(0.0)
    end
end

@inline function W_shape(::Val{3}, y::T)::T where {T<:AbstractFloat}
    y = abs(y)
    if y <= T(1.0)
        T(2.0) / T(3.0) - y^2 + y^3 / T(2.0)
    elseif y <= T(2.0)
        (T(2.0) - y)^3 / T(6.0)
    else
        T(0.0)
    end
end

@inline function W_shape(::Val{4}, y::T)::T where {T<:AbstractFloat}
    y = abs(y)
    if y <= T(0.5)
        T(115.0) / T(192.0) - T(5.0) * y^2 / T(8.0) + y^4 / T(4.0)
    elseif y <= T(1.5)
        (T(55.0) + T(20.0) * y - T(120.0) * y^2 + T(80.0) * y^3 - T(16.0) * y^4) / T(96.0)
    elseif y < T(2.5)
        (T(5.0) - T(2.0) * y)^4 / T(384.0)
    else
        T(0.0)
    end
end

@inline function W_shape(::Val{5}, y::T)::T where {T<:AbstractFloat}
    y = abs(y)
    if y <= T(1.0)
        T(11.0) / T(20.0) - y^2 / T(2.0) + y^4 / T(4.0) - y^5 / T(12.0)
    elseif y <= T(2.0)
        T(17.0) / T(40.0) + T(5.0) * y / T(8.0) - T(7.0) * y^2 / T(4.0) +
        T(5.0) * y^3 / T(4.0) - T(3.0) * y^4 / T(8.0) + y^5 / T(24.0)
    elseif y < T(3.0)
        (T(3.0) - y)^5 / T(120.0)
    else
        T(0.0)
    end
end

@inline Shape(::Val{Order}, y::AbstractFloat) where {Order} = W_shape(Val(Order - 1), y)

"""One-sided integer bound for shape support: cells touched per dimension."""
static_bound(::Val{Order}) where {Order} = Int64(ceil(Order / 2))

"""Half width of stencil with dilation factor."""
stencil_half_width(::Val{Order}, factor::Real) where {Order} = static_bound(Val(Order)) * factor

"""Full integer loop range for deposition / interpolation stencil."""
function stencil_range(::Val{Order}, factor::Real) where {Order}
    b = Int64(stencil_half_width(Val(Order), factor))
    (-b):(b + 1)
end
