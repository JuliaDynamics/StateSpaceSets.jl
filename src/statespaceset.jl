using StaticArraysCore, LinearAlgebra
using Base.Iterators: flatten
using Statistics

export AbstractStateSpaceSet, minima, maxima
export SVector, SMatrix
export minmaxima, columns, standardize, dimension
export cov, cor, mean_and_cov

# D = dimension, T = element type, V = container type
# note that the container type is given as keyword `container` to
# all functions that somehow end up making a state space set.
abstract type AbstractStateSpaceSet{D, T, V} <: AbstractVector{V} end

# Core extensions and functions:
"""
    dimension(thing) -> D

Return the dimension of the `thing`, in the sense of state-space dimensionality.
"""
dimension(::AbstractStateSpaceSet{D}) where {D} = D
Base.eltype(::AbstractStateSpaceSet{D,T}) where {D,T} = T
Base.vec(X::AbstractStateSpaceSet) = X.data
containertype(::AbstractStateSpaceSet{D,T,V}) where {D,T,V} = V

###########################################################################################
# Base extensions
###########################################################################################
for f in (
        :length, :sort!, :iterate, :eachindex, :eachrow, :firstindex, :lastindex, :size,
    )
    @eval Base.$(f)(d::AbstractStateSpaceSet, args...; kwargs...) = $(f)(vec(d), args...; kwargs...)
end

Base.:(==)(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = vec(d1) == vec(d2)
Base.copy(d::AbstractStateSpaceSet) = typeof(d)(copy(vec(d)))
Base.sort(d::AbstractStateSpaceSet) = sort!(copy(d))
@inline Base.eltype(::Type{<:AbstractStateSpaceSet{D, T}}) where {D, T} = SVector{D, T}
@inline Base.IteratorSize(::Type{<:AbstractStateSpaceSet}) = Base.HasLength()
Base.eachcol(ds::AbstractStateSpaceSet) = (ds[:, i] for i in 1:dimension(ds))

"""
    columns(ssset) -> x, y, z, ...

Return the individual columns of the state space set allocated as `Vector`s.
Equivalent with `collect(eachcol(ssset))`.
"""
function columns end
@generated function columns(data::AbstractStateSpaceSet{D}) where {D}
    gens = [:(data[:, $k]) for k=1:D]
    quote tuple($(gens...)) end
end

###########################################################################################
# Indexing
###########################################################################################
# 1D indexing over the container elements:
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int) = vec(d)[i]
@inline Base.getindex(d::AbstractStateSpaceSet, i) = StateSpaceSet(vec(d)[i])
@inline Base.lastindex(d::AbstractStateSpaceSet) = length(d)

# 2D indexing with second index being column (reduces indexing to 1D indexing)
@inline Base.getindex(d::AbstractStateSpaceSet, i, ::Colon) = d[i]

# 2D indexing where dataset behaves as a matrix
# with each column a dynamic variable timeseries
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int, j::Int) = vec(d)[i][j]
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, j::Int) =
[vec(d)[k][j] for k in eachindex(d)]
@inline Base.getindex(d::AbstractStateSpaceSet, i::AbstractVector, j::Int) =
[vec(d)[k][j] for k in i]
@inline Base.getindex(d::AbstractStateSpaceSet, i::Int, j::AbstractVector) = d[i][j]
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, ::Colon) = d
@inline Base.getindex(d::AbstractStateSpaceSet, ::Colon, v::AbstractVector) =
StateSpaceSet([d[i][v] for i in eachindex(d)])
@inline Base.getindex(d::AbstractStateSpaceSet, v1::AbstractVector, v::AbstractVector) =
StateSpaceSet([d[i][v] for i in v1])

# Set index stuff
@inline Base.setindex!(d::AbstractStateSpaceSet, v, i::Int) = (vec(d)[i] = v)

function Base.dotview(d::AbstractStateSpaceSet, ::Colon, ::Int)
    error("`setindex!` is not defined for StateSpaceSets and the given arguments. "*
    "Best to create a new dataset or `Vector{SVector}` instead of in-place operations.")
end

###########################################################################
# Appending
###########################################################################
Base.append!(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = (append!(vec(d1), vec(d2)); d1)
Base.push!(d::AbstractStateSpaceSet, new_item) = (push!(vec(d), new_item); d)

function Base.hcat(d::AbstractStateSpaceSet{D, T, V}, x::AbstractVector{<:Real}) where {D, T, V}
    L = length(d)
    L == length(x) || error("statespaceset and vector must be of same length")
    if V == SVector{D, T}
        V2 = SVector{D+1, T}
    else
        V2 = V # it is `Vector{T}` instead
    end
    data = Vector{V2}(undef, L)
    @inbounds for i in 1:L
        if V == SVector{D, T}
            e = V2(d[i]..., x[i])
        else
            e = vcat(d[i], x[i])
        end
        data[i] = e
    end
    return StateSpaceSet(data)
end

function Base.hcat(x::AbstractVector{<:Real}, d::AbstractStateSpaceSet{D, T, V}) where {D, T, V}
    L = length(d)
    L == length(x) || error("statespaceset and vector must be of same length")
    if V == SVector{D, T}
        V2 = SVector{D+1, T}
    else
        V2 = V # it is `Vector{T}` instead
    end
    data = Vector{V2}(undef, L)
    @inbounds for i in 1:L
        if V <: SVector
            e = V2(x[i], d[i]...)
        else
            e = vcat(x[i], d[i])
        end
        data[i] = e
    end
    return StateSpaceSet(data)
end

function Base.hcat(ds::AbstractStateSpaceSet{<: Any, T}...) where {T}
    Ls = length.(ds)
    maxlen = maximum(Ls)
    all(Ls .== maxlen) || error("StateSpaceSets must be of same length")
    V = containertype(first(ds))
    if V <: SVector
        newdim = sum(dimension.(ds))
        V2 = SVector{newdim, T}
    else
        V2 = V # it is `Vector`
    end
    v = Vector{V}(undef, maxlen)
    for i = 1:maxlen
        v[i] = V(collect(Iterators.flatten(d[i] for d in ds)))
    end
    return StateSpaceSet(v)
end

# TODO: This can probably be done more efficiently by pre-computing the size of the
# `SVector`s, and doing something similar to the explicit two-argument versions above.
# However, it's not immediately clear how to do this efficiently. This implementation
# converts every input to a dataset first and promotes everything to a common type.
# It's not optimal, because it allocates unnecessarily, but it works.
# If this method is made more efficient, the method above can be dropped.
function Base.hcat(xs::Union{AbstractVector{<:Real}, AbstractStateSpaceSet}...)
    ds = StateSpaceSet.(xs)
    Ls = length.(ds)
    maxlen = maximum(Ls)
    all(Ls .== maxlen) || error("StateSpaceSets must be of same length")
    V = typeof(xs[1][1])
    newdim = sum(dimension.(ds))
    T = promote_type(eltype.(ds)...)
    if V <: SVector
        V2 = SVector{newdim, T}
    else
        V2 = V # it is `Vector`
    end
    v = Vector{v}(undef, maxlen)
    for i = 1:maxlen
        if V <: SVector
            e = V(Iterators.flatten(d[i] for d in xs)...,)
        else
            e = V(collect(Iterators.flatten(d[i] for d in xs)))
        end
        v[i] = e
    end
    return StateSpaceSet(v)
end

#####################################################################################
#                                   Pretty Printing                                 #
#####################################################################################
function Base.summary(d::AbstractStateSpaceSet{D, T}) where {D, T}
    N = length(d)
    return "$D-dimensional $(nameof(typeof(d))){$(T)} with $N points"
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
