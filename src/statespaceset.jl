using StaticArraysCore, LinearAlgebra
using Base.Iterators: flatten

export StateSpaceSet, AbstractStateSpaceSet, minima, maxima
export SVector, SMatrix
export minmaxima, columns, standardize, dimension

abstract type AbstractStateSpaceSet{D, T} end

# Core extensions and functions:
"""
    dimension(thing) -> D
Return the dimension of the `thing`, in the sense of state-space dimensionality.
"""
dimension(::AbstractStateSpaceSet{D,T}) where {D,T} = D
Base.eltype(::AbstractStateSpaceSet{D,T}) where {D,T} = T
Base.parent(X::AbstractStateSpaceSet{D,T}) where {D,T} = X.data
@inline Base.size(d::AbstractStateSpaceSet{D,T}) where {D,T} = (length(d.data), D)
@inline Base.size(d::AbstractStateSpaceSet, i) = size(d)[i]

"""
    columns(dataset) -> x, y, z, ...
Return the individual columns of the dataset.
"""
function columns end
@generated function columns(data::AbstractStateSpaceSet{D, T}) where {D, T}
    gens = [:(data[:, $k]) for k=1:D]
    quote tuple($(gens...)) end
end
columns(x::AbstractVector{<:Real}) = (x, )

###########################################################################################
# Base extensions
###########################################################################################
for f in (
        :length, :vec, :sort!, :IteratorSize, :iterate, :eachindex, :eachrow,
        :firstindex,
    )
    @eval Base.$(f)(d::AbstractStateSpaceSet, args...) = $(f)(parent(d), args...)
end

Base.:(==)(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = d1.data == d2.data
Base.copy(d::AbstractStateSpaceSet) = typeof(d)(copy(d.data))
Base.sort(d::AbstractStateSpaceSet) = sort!(copy(d))
@inline Base.eltype(::Type{<:AbstractStateSpaceSet{D, T}}) where {D, T} = SVector{D, T}
Base.eachcol(ds::AbstractStateSpaceSet) = (ds[:, i] for i in 1:size(ds, 2))

###########################################################################################
# Indexing
###########################################################################################
# 1D indexing over the container elements:
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int) = d.data[i]
@inline Base.getindex(d::AbstractStateSpaceSet, i) = StateSpaceSet(d.data[i])
@inline Base.lastindex(d::AbstractStateSpaceSet) = length(d)
@inline Base.lastindex(d::AbstractStateSpaceSet, k) = size(d)[k]

# 2D indexing with second index being column (reduces indexing to 1D indexing)
@inline Base.getindex(d::AbstractStateSpaceSet, i, ::Colon) = d[i]

# 2D indexing where dataset behaves as a matrix
# with each column a dynamic variable timeseries
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int, j::Int) = d.data[i][j]
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, j::Int) =
[d.data[k][j] for k in eachindex(d)]
@inline Base.getindex(d::AbstractStateSpaceSet, i::AbstractVector, j::Int) =
[d.data[k][j] for k in i]
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int, j::AbstractVector) = d[i][j]
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, ::Colon) = d
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, v::AbstractVector) =
StateSpaceSet([d[i][v] for i in eachindex(d)])
@inline Base.getindex(d::AbstractStateSpaceSet, v1::AbstractVector, v::AbstractVector) =
StateSpaceSet([d[i][v] for i in v1])

# Set index stuff
@inline Base.setindex!(d::AbstractStateSpaceSet, v, i::Int) = (d.data[i] = v)

function Base.dotview(d::AbstractStateSpaceSet, ::Colon, ::Int)
    error("`setindex!` is not defined for Datasets and the given arguments. "*
    "Best to create a new dataset or `Vector{SVector}` instead of in-place operations.")
end

###########################################################################
# Appending
###########################################################################
Base.append!(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = (append!(vec(d1), vec(d2)); d1)
Base.push!(d::AbstractStateSpaceSet, new_item) = (push!(d.data, new_item); d)

function Base.hcat(d::AbstractStateSpaceSet{D, T}, x::Vector{<:Real}) where {D, T}
    L = length(d)
    L == length(x) || error("dataset and vector must be of same length")
    data = Vector{SVector{D+1, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = SVector{D+1, T}(d[i]..., x[i])
    end
    return StateSpaceSet(data)
end

function Base.hcat(x::Vector{<:Real}, d::AbstractStateSpaceSet{D, T}) where {D, T}
    L = length(d)
    L == length(x) || error("dataset and vector must be of same length")
    data = Vector{SVector{D+1, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = SVector{D+1, T}(x[i], d[i]...)
    end
    return StateSpaceSet(data)
end

function Base.hcat(ds::Vararg{AbstractStateSpaceSet{D, T} where {D}, N}) where {T, N}
    Ls = length.(ds)
    maxlen = maximum(Ls)
    all(Ls .== maxlen) || error("Datasets must be of same length")
    newdim = sum(dimension.(ds))
    v = Vector{SVector{newdim, T}}(undef, maxlen)
    for i = 1:maxlen
        v[i] = SVector{newdim, T}(Iterators.flatten(ds[d][i] for d = 1:N)...,)
    end
    return StateSpaceSet(v)
end

# TODO: This can probably be done more efficiently by pre-computing the size of the
# `SVector`s, and doing something similar to the explicit two-argument versions above.
# However, it's not immediately clear how to do this efficiently. This implementation
# converts every input to a dataset first and promotes everything to a common type.
# It's not optimal, because it allocates unnecessarily, but it works.
# If this method is made more efficient, the method above can be dropped.
function hcat(xs::Vararg{Union{AbstractVector{<:Real}, AbstractStateSpaceSet{D, T} where {D, T}}, N}) where {N}
    ds = StateSpaceSet.(xs)
    Ls = length.(ds)
    maxlen = maximum(Ls)
    all(Ls .== maxlen) || error("Datasets must be of same length")
    newdim = sum(dimension.(ds))
    T = promote_type(eltype.(ds)...)
    v = Vector{SVector{newdim, T}}(undef, maxlen)
    for i = 1:maxlen
        v[i] = SVector{newdim, T}(Iterators.flatten(ds[d][i] for d = 1:N)...,)
    end
    return StateSpaceSet(v)
end

#####################################################################################
#                                StateSpaceSet <-> Matrix                                 #
#####################################################################################
function Base.Matrix{S}(d::AbstractStateSpaceSet{D,T}) where {S, D, T}
    mat = Matrix{S}(undef, length(d), D)
    for j in 1:D
        for i in 1:length(d)
            @inbounds mat[i,j] = d.data[i][j]
        end
    end
    mat
end
Base.Matrix(d::AbstractStateSpaceSet{D,T}) where {D, T} = Matrix{T}(d)

function StateSpaceSet(mat::AbstractMatrix{T}; warn = true) where {T}
    N, D = size(mat)
    warn && D > 100 && @warn "You are attempting to make a StateSpaceSet of dimensions > 100"
    warn && D > N && @warn "You are attempting to make a StateSpaceSet of a matrix with more columns than rows."
    StateSpaceSet{D,T}(reshape(reinterpret(SVector{D,T}, vec(transpose(mat))), (N,)))
end

#####################################################################################
#                                   Pretty Printing                                 #
#####################################################################################
function Base.summary(d::StateSpaceSet{D, T}) where {D, T}
    N = length(d)
    return "$D-dimensional StateSpaceSet{$(T)} with $N points"
end

function matstring(d::AbstractStateSpaceSet{D, T}) where {D, T}
    N = length(d)
    if N > 50
        mat = zeros(eltype(d), 50, D)
        for (i, a) in enumerate(flatten((1:25, N-24:N)))
            mat[i, :] .= d[a]
        end
    else
        mat = Matrix(d)
    end
    s = sprint(io -> show(IOContext(io, :limit=>true), MIME"text/plain"(), mat))
    s = join(split(s, '\n')[2:end], '\n')
    tos = summary(d)*"\n"*s
    return tos
end

Base.show(io::IO, ::MIME"text/plain", d::AbstractStateSpaceSet) = print(io, matstring(d))
Base.show(io::IO, d::AbstractStateSpaceSet) = print(io, summary(d))

#####################################################################################
#                                 Minima and Maxima                                 #
#####################################################################################
"""
    minima(dataset)
Return an `SVector` that contains the minimum elements of each timeseries of the
dataset.
"""
function minima(data::AbstractStateSpaceSet{D, T}) where {D, T<:Real}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] < m[i]
                m[i] = point[i]
            end
        end
    end
    return SVector{D,T}(m)
end

"""
    maxima(dataset)
Return an `SVector` that contains the maximum elements of each timeseries of the
dataset.
"""
function maxima(data::AbstractStateSpaceSet{D, T}) where {D, T<:Real}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > m[i]
                m[i] = point[i]
            end
        end
    end
    return SVector{D, T}(m)
end

"""
    minmaxima(dataset)
Return `minima(dataset), maxima(dataset)` without doing the computation twice.
"""
function minmaxima(data::AbstractStateSpaceSet{D, T}) where {D, T<:Real}
    mi = Vector(data[1])
    ma = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > ma[i]
                ma[i] = point[i]
            elseif point[i] < mi[i]
                mi[i] = point[i]
            end
        end
    end
    return SVector{D, T}(mi), SVector{D, T}(ma)
end

#####################################################################################
#                                     SVD                                           #
#####################################################################################
using LinearAlgebra
# SVD of Base seems to be much faster when the "long" dimension of the matrix
# is the first one, probably due to Julia's column major structure.
# This does not depend on using `svd` or `svdfact`, both give same timings.
# In fact it is so much faster, that it is *much* more worth it to
# use `Matrix(data)` instead of `reinterpret` in order to preserve the
# long dimension being the first.
"""
    svd(d::AbstractStateSpaceSet) -> U, S, Vtr
Perform singular value decomposition on the dataset.
"""
function LinearAlgebra.svd(d::AbstractStateSpaceSet)
    F = svd(Matrix(d))
    return F[:U], F[:S], F[:Vt]
end

#####################################################################################
#                                standardize                                         #
#####################################################################################
using Statistics: mean, std

"""
    standardize(d::StateSpaceSet) → r

Create a standardized version of the input dataset where each timeseries (column)
is transformed to have mean 0 and standard deviation 1.
"""
standardize(d::AbstractStateSpaceSet) = StateSpaceSet(standardized_timeseries(d)[1]...)
function standardized_timeseries(d::AbstractStateSpaceSet)
    xs = columns(d)
    means = mean.(xs)
    stds = std.(xs)
    for i in eachindex(xs)
        xs[i] .= (xs[i] .- means[i]) ./ stds[i]
    end
    return xs, means, stds
end

###########################################################################
# Concrete implementation
###########################################################################
"""
    StateSpaceSet{D, T} <: AbstractStateSpaceSet{D,T}

A dedicated interface for sets in a state space.
It is an **ordered container of equally-sized points** of length `D`.
Each point is represented by `SVector{D, T}`.
The data are a standard Julia `Vector{SVector}`, and can be obtained with
`parent(ssset::StateSpaceSet)`.
Typically the order of points in the set is the time direction, but it doesn't have to be.

When indexed with 1 index, `StateSpaceSet` is like a vector of points.
When indexed with 2 indices it behaves like a matrix that has each of the columns be the
timeseries of each of the variables.
When iterated over, it iterates over its contained points.
See description of indexing below for more.

`StateSpaceSet` also supports almost all sensible vector operations like
`append!, push!, hcat, eachrow`,
among others.

## Description of indexing

In the following let `i, j` be integers, `typeof(X) <: AbstractStateSpaceSet`
and `v1, v2` be `<: AbstractVector{Int}` (`v1, v2` could also be ranges,
and for performance benefits make `v2` an `SVector{Int}`).

* `X[i] == X[i, :]` gives the `i`th point (returns an `SVector`)
* `X[v1] == X[v1, :]`, returns a `StateSpaceSet` with the points in those indices.
* `X[:, j]` gives the `j`th variable timeseries (or collection), as `Vector`
* `X[v1, v2], X[:, v2]` returns a `StateSpaceSet` with the appropriate entries (first indices
  being "time"/point index, while second being variables)
* `X[i, j]` value of the `j`th variable, at the `i`th timepoint

Use `Matrix(ssset)` or `StateSpaceSet(matrix)` to convert. It is assumed
that each *column* of the `matrix` is one variable.
If you have various timeseries vectors `x, y, z, ...` pass them like
`StateSpaceSet(x, y, z, ...)`. You can use `columns(dataset)` to obtain the reverse,
i.e. all columns of the dataset in a tuple.
"""
struct StateSpaceSet{D, T} <: AbstractStateSpaceSet{D,T}
    data::Vector{SVector{D,T}}
end
# Empty dataset:
StateSpaceSet{D, T}() where {D,T} = StateSpaceSet(SVector{D,T}[])

# Identity constructor:
StateSpaceSet{D, T}(s::StateSpaceSet{D, T}) where {D,T} = s
StateSpaceSet(s::StateSpaceSet) = s

###########################################################################
# StateSpaceSet(Vectors of stuff)
###########################################################################
StateSpaceSet(s::AbstractVector{T}) where {T} = StateSpaceSet(SVector.(s))

function StateSpaceSet(v::Vector{<:AbstractArray{T}}) where {T<:Number}
    D = length(v[1])
    @assert length(unique!(length.(v))) == 1 "All input vectors must have same length"
    D > 100 && @warn "You are attempting to make a StateSpaceSet of dimensions > 100"
    L = length(v)
    data = Vector{SVector{D, T}}(undef, L)
    for i in 1:length(v)
        D != length(v[i]) && throw(ArgumentError(
        "All data-points in a StateSpaceSet must have same size"
        ))
        @inbounds data[i] = SVector{D,T}(v[i])
    end
    return StateSpaceSet{D, T}(data)
end

@generated function _dataset(vecs::Vararg{<:AbstractVector{T},D}) where {D, T}
    gens = [:(vecs[$k][i]) for k=1:D]
    D > 100 && @warn "You are attempting to make a StateSpaceSet of dimensions > 100"
    quote
        L = typemax(Int)
        for x in vecs
            l = length(x)
            l < L && (L = l)
        end
        data = Vector{SVector{$D, T}}(undef, L)
        for i in 1:L
            @inbounds data[i] = SVector{$D, T}($(gens...))
        end
        data
    end
end

function StateSpaceSet(vecs::Vararg{<:AbstractVector{T}}) where {T}
    return StateSpaceSet(_dataset(vecs...))
end

StateSpaceSet(xs::Vararg{Union{AbstractVector, AbstractStateSpaceSet}}) = hcat(xs...)
StateSpaceSet(x::Vector{<:Real}, y::AbstractStateSpaceSet{D, T}) where {D, T} = hcat(x, y)
StateSpaceSet(x::AbstractStateSpaceSet{D, T}, y::Vector{<:Real}) where {D, T} = hcat(x, y)
