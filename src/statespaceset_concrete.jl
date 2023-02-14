export StateSpaceSet

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

###########################################################################
# View
###########################################################################
"""
    SubStateSpaceSet{D, T, <:AbstractStateSpaceSet{D,T}, <:SubArray{<:SVector{D,T},1}} <: AbstractStateSpaceSet{D,T}

A view of an `AbstractStateSpaceSet`, as returned by the `view` function
or the `@view` macro on a dataset instance. A `SubStateSpaceSet` is an `AbstractStateSpaceSet`
of the same type as its parent, so indexing, iteration, and most other functions
can be expected to work in the same way for both the parent and the view.
"""
struct SubStateSpaceSet{D, T, P<:AbstractStateSpaceSet{D,T}, S<:SubArray{<:SVector{D,T},1}} <: AbstractStateSpaceSet{D,T}
    parent::P
    data::S
    function SubStateSpaceSet(par, data)
        @assert parent(data) === par.data
        P = typeof(par)
        S = typeof(data)
        SV = eltype(P)
        T = eltype(SV)
        D = length(SV)
        new{D,T,P,S}(par, data)
    end
end

function Base.summary(sd::SubStateSpaceSet{D, T}) where {D, T}
    N = length(sd)
    return "$N-element view of $D-dimensional StateSpaceSet{$(T)}"
end

Base.parent(sd::SubStateSpaceSet) = sd.parent
Base.parentindices(sd::SubStateSpaceSet) = parentindices(sd.data)

"""
    view(d::StateSpaceSet, indices)

Return a view into the parent dataset `d`, as a [`SubStateSpaceSet`](@ref)
that contains the datapoints of `d` referred to by `indices`.
"""
Base.view(d::AbstractStateSpaceSet, i) = SubStateSpaceSet(d, view(d.data, i))

function Base.view(::AbstractStateSpaceSet, ::Any, ::Any, ::Vararg)
    throw(ArgumentError("StateSpaceSet views only accept indices on one dimension"))
end
