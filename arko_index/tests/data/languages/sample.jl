"""
Geometry helpers.
"""
module Geometry

using LinearAlgebra
export Point, distance, centroid

# A point in the plane.
struct Point{T<:Real}
    x::T
    y::T
end

abstract type Shape end

mutable struct Polygon <: Shape
    points::Vector{Point{Float64}}
end

"""
    distance(a, b)

The Euclidean distance between `a` and `b`.
"""
function distance(a::Point, b::Point)
    return sqrt((a.x - b.x)^2 + (a.y - b.y)^2)
end

centroid(ps::Vector{<:Point}) = Point(sum(p.x for p in ps) / length(ps), sum(p.y for p in ps) / length(ps))

function area(p::Polygon)
    n = length(p.points)
    s = 0.0
    for i in 1:n
        a, b = p.points[i], p.points[mod1(i + 1, n)]
        s += a.x * b.y - b.x * a.y
    end
    abs(s) / 2
end

macro twice(ex)
    quote
        $(esc(ex))
        $(esc(ex))
    end
end

end # module
