
#####################################################################################
#                                 Minima and Maxima                                 #
#####################################################################################
"""
    minima(dataset)

Return an `SVector` that contains the minimum elements of each timeseries of the
dataset.
"""
function minima(data::AbstractStateSpaceSet{D, T, V}) where {D, T<:Real, V}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] < m[i]
                m[i] = point[i]
            end
        end
    end
    return V(m)
end

"""
    maxima(dataset)
Return an `SVector` that contains the maximum elements of each timeseries of the
dataset.
"""
function maxima(data::AbstractStateSpaceSet{D, T, V}) where {D, T<:Real, V}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > m[i]
                m[i] = point[i]
            end
        end
    end
    return V(m)
end

"""
    minmaxima(dataset)
Return `minima(dataset), maxima(dataset)` without doing the computation twice.
"""
function minmaxima(data::AbstractStateSpaceSet{D, T, V}) where {D, T<:Real, V}
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
    return V(mi), V(ma)
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
cov(x::AbstractStateSpaceSet) = fastcov(vec(x))

"""
    mean_and_cov(d::StateSpaceSet) → μ, m::SMatrix

Return a tuple of the column means `μ` and covariance matrix `m`.

Column means are always computed for the covariance matrix, so this is faster
than computing both quantities separately.
"""
mean_and_cov(x::AbstractStateSpaceSet) = fastmean_and_cov(vec(x))

"""
    cor(d::StateSpaceSet) → m::SMatrix

Compute the corrlation matrix `m` from the columns of `d`, where `m[i, j]` is the
correlation between `d[:, i]` and `d[:, j]`.
"""
cor(x::AbstractStateSpaceSet) = fastcor(vec(x))

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