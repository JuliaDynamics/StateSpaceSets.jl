using StaticArraysCore, LinearAlgebra
using Base.Iterators: flatten
using Statistics

export AbstractStateSpaceSet, minima, maxima
export SVector, SMatrix
export minmaxima, columns, standardize, dimension
export cov, cor, mean_and_cov

abstract type AbstractStateSpaceSet{D, T} end

# Core extensions and functions:
"""
    dimension(thing) -> D
Return the dimension of the `thing`, in the sense of state-space dimensionality.
"""
dimension(::AbstractStateSpaceSet{D,T}) where {D,T} = D
Base.eltype(::AbstractStateSpaceSet{D,T}) where {D,T} = T
Base.vec(X::AbstractStateSpaceSet{D,T}) where {D,T} = X.data

# TODO: This will break once we make SSS <: AbstractVector
@inline Base.size(d::AbstractStateSpaceSet{D,T}) where {D,T} = (length(vec(d)), D)
@inline Base.size(d::AbstractStateSpaceSet, i) = size(d)[i]

###########################################################################################
# Base extensions
###########################################################################################
for f in (
        :length, :sort!, :iterate, :eachindex, :eachrow, :firstindex,
    )
    @eval Base.$(f)(d::AbstractStateSpaceSet, args...; kwargs...) = $(f)(vec(d), args...; kwargs...)
end

Base.:(==)(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = d1.data == d2.data
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
@generated function columns(data::AbstractStateSpaceSet{D, T}) where {D, T}
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
    error("`setindex!` is not defined for Datasets and the given arguments. "*
    "Best to create a new dataset or `Vector{SVector}` instead of in-place operations.")
end

###########################################################################
# Appending
###########################################################################
Base.append!(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet) = (append!(vec(d1), vec(d2)); d1)
Base.push!(d::AbstractStateSpaceSet, new_item) = (push!(vec(d), new_item); d)

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
#                                standardize                                        #
#####################################################################################
using Statistics: mean, std

"""
    standardize(d::StateSpaceSet) → r

Create a standardized version of the input set where each column
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


#####################################################################################
#                          covariance/correlation matrix                            #
#####################################################################################
import Statistics: cov, cor
using Statistics: mean, std
using StaticArraysCore: MMatrix, MVector, SMatrix, SVector

"""
    cov(d::StateSpaceSet) → m::SMatrix

Compute the covariance matrix `m` from the columns of `d`, where `m[i, j]` is the covariance
between `d[:, i]` and `d[:, j]`.
"""
cov(x::AbstractStateSpaceSet) = fastcov(x.data)

"""
    mean_and_cov(d::StateSpaceSet) → μ, m::SMatrix

Return a tuple of the column means `μ` and covariance matrix `m`. 

Column means are always computed for the covariance matrix, so this is faster 
than computing both quantities separately.
"""
mean_and_cov(x::AbstractStateSpaceSet) = fastmean_and_cov(x.data)

"""
    cor(d::StateSpaceSet) → m::SMatrix

Compute the corrlation matrix `m` from the columns of `d`, where `m[i, j]` is the 
correlation between `d[:, i]` and `d[:, j]`.
"""
cor(x::AbstractStateSpaceSet) = fastcor(x.data)

function fastcov(x::Vector{SVector{D, T}}) where {D, T}
    T <: AbstractFloat || error("Need `eltype(x[i]) <: AbstractFloat` ∀ i ∈ 1:length(x). Got `eltype(x[i])=$(eltype(first(x)))`")
    μ = mean(x)
    return fastcov(μ, x)
end

function fastcov(μ, x::Vector{SVector{D, T}}) where {D, T}
    T <: AbstractFloat || error("Need `eltype(x[i]) <: AbstractFloat` ∀ i ∈ 1:length(x). Got `eltype(x[i])=$(eltype(first(x)))`")
    N = length(x) - 1
    C = MMatrix{D, D}(zeros(D, D))
    x̄ = mean(x)
    Δx = MVector{D}(zeros(D))
    @inbounds for xᵢ in x
        Δx .= xᵢ - μ
        C .+= Δx * transpose(Δx)
    end
    C ./= N
    return SMatrix{D, D}(C)
end

function fastmean_and_cov(x::Vector{SVector{D, T}}) where {D, T}
    μ = mean(x)
    Σ = fastcov(μ, x)
    return μ, Σ
end

# Non-allocating and faster than writing a wrapper.
function fastcor(x::Vector{SVector{D, T}}) where {D, T}
    μ, Σ = fastmean_and_cov(x)
    σ = std(x)
    C = MMatrix{D, D}(zeros(D, D))
    for j in 1:D
        for i in 1:D
            C[i, j] = Σ[i, j] / (σ[i] * σ[j])
        end
    end
    return SMatrix{D, D}(C)
end