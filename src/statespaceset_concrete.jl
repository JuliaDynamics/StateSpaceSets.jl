export StateSpaceSet, SSSet

"""
    StateSpaceSet{D, T, V} <: AbstractVector{V}

A dedicated interface for sets in a state space.
It is an **ordered container of equally-sized points** of length `D`,
with element type `T`,
represented by a vector of type `V`. Typically `V` is `SVector{D,T}` or `Vector{T}`
and the data are always stored internally as `Vector{V}`.
`SSSet` is an alias for `StateSpaceSet`.

The underlying `Vector{V}` can be obtained by `vec(ssset)`, although this is almost
never necessary because `StateSpaceSet` subtypes `AbstractVector` and extends its interface.
`StateSpaceSet` also supports almost all sensible vector operations like
`append!, push!, hcat, eachrow`, among others.
When iterated over, it iterates over its contained points.

## Construction

Constructing a `StateSpaceSet` is done in three ways:

1. By giving in each individual **columns** of the state space set as `Vector{<:Real}`:
   `StateSpaceSet(x, y, z, ...)`.
2. By giving in a matrix whose rows are the state space points: `StateSpaceSet(m)`.
3. By giving in directly a vector of vectors (state space points): `StateSpaceSet(v_of_v)`.

All constructors allow for two keywords:
- `container` which sets the type of `V` (the type of inner vectors).
  At the moment options are only `SVector`, `MVector`, or `Vector`, and by default `SVector` is used.
- `names` which can be an iterable of length `D` whose elements are `Symbol`s.
  This allows assigning a name to each dimension and accessing the dimension by name,
  see below. `names` is `nothing` if not given. Use `StateSpaceSet(s; names)` to add
  names to an existing set `s`.

## Description of indexing

When indexed with 1 index, `StateSpaceSet` behaves exactly like its encapsulated vector.
i.e., a vector of vectors (state space points).
When indexed with 2 indices it behaves like a matrix where each row is a point.

In the following let `i, j` be integers, `typeof(X) <: AbstractStateSpaceSet`
and `v1, v2` be `<: AbstractVector{Int}` (`v1, v2` could also be ranges,
and for performance benefits make `v2` an `SVector{Int}`).

* `X[i] == X[i, :]` gives the `i`th point (returns an `SVector`)
* `X[v1] == X[v1, :]`, returns a `StateSpaceSet` with the points in those indices.
* `X[:, j]` gives the `j`th variable timeseries (or collection), as `Vector`
* `X[v1, v2], X[:, v2]` returns a `StateSpaceSet` with the appropriate entries (first indices
  being "time"/point index, while second being variables)
* `X[i, j]` value of the `j`th variable, at the `i`th timepoint

In all examples above, `j` can also be a `Symbol`, provided that `names` has been
given when creating the state space set. This allows accessing a dimension by name.
This is provided as a convenience and it is not an optimized operation, hence
recommended to be used primarily with `X[:, j::Symbol]`.

Use `Matrix(ssset)` or `StateSpaceSet(matrix)` to convert. It is assumed
that each *column* of the `matrix` is one variable.
If you have various timeseries vectors `x, y, z, ...` pass them like
`StateSpaceSet(x, y, z, ...)`. You can use `columns(dataset)` to obtain the reverse,
i.e. all columns of the dataset in a tuple.
"""
struct StateSpaceSet{D, T, V<:AbstractVector, N} <: AbstractStateSpaceSet{D,T,V,N}
    data::Vector{V}
    names::N
    function StateSpaceSet{D, T, V, N}(data, names) where {D,T,V,N}
        if !isnothing(names)
            if length(names) != D
                error("Given names must be as many as the dimension of the set!")
            end
        end
        return new{D, T, V, N}(data, names)
    end
end
const SSSet = StateSpaceSet # alias
# Empty dataset:
StateSpaceSet{D, T}(; names = nothing) where {D,T} = StateSpaceSet{D,T,SVector{D,T},typeof(names)}(SVector{D,T}[], names)

# Identity constructor:
StateSpaceSet{D, T}(s::StateSpaceSet{D, T}) where {D,T} = s
StateSpaceSet(s::StateSpaceSet; names = nothing) = StateSpaceSet(vec(s); names)

function StateSpaceSet(v::Vector{V}; container = SVector, names = nothing) where {V<:AbstractVector}
    n = length(v[1])
    t = eltype(v[1])
    for p in v
        length(p) != n && error("Inner vectors must all have same length")
    end
    # TODO: There must be a way to generalize this to any container!
    # we can use `Base.typename(typeof(v[1])).wrapper` but this is so internal... :(
    if container <: SVector
        U = SVector{n, t}
    elseif container <: MVector
        U = MVector{n, t}
    else
        U = Vector{t}
    end
    if U != V
        u = U.(v)
    else
        u = v
    end
    return StateSpaceSet{n,t,U,typeof(names)}(u, names)
end

# Concatenating existing state space sets
function StateSpaceSet(xs::AbstractStateSpaceSet...)
    return hcat(xs...)
end

###########################################################################
# StateSpaceSet(Vectors of stuff)
###########################################################################
function StateSpaceSet(vecs::AbstractVector{T}...; container = SVector, names = nothing) where {T}
    data = _ssset(vecs...)
    if container != SVector
        data = container.(data)
    end
    D = length(vecs)
    V = typeof(data[1])
    return StateSpaceSet{D,T,V,typeof(names)}(data, names)
end

@generated function _ssset(vecs::AbstractVector{T}...) where {T}
    D = length(vecs)
    gens = [:(vecs[$k][i]) for k=1:D]
    quote
        # we can't use a generator inside a `@generated` function, it's not "pure"
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

function StateSpaceSet(mat::AbstractMatrix{T}; warn = true, container = SVector, names = nothing) where {T}
    N, D = size(mat)
    warn && D > 100 && @warn "You are attempting to make a StateSpaceSet of dimensions > 100"
    warn && D > N && @warn "You are attempting to make a StateSpaceSet of a matrix with more columns than rows."
    if container <: SVector
        V = SVector{D,T}
    elseif container <: MVector
        V = MVector{D,T}
    else
        V = Vector{T}
    end
    data = [V(row) for row in eachrow(mat)]
    StateSpaceSet{D,T,V,typeof(names)}(data, names)
end

###########################################################################
# View
###########################################################################
"""
    SubStateSpaceSet

A view of an `AbstractStateSpaceSet`, as returned by the `view` function
or the `@view` macro on a statespaceset instance. A `SubStateSpaceSet` is an `AbstractStateSpaceSet`
of the same type as its parent, so indexing, iteration, and most other functions
can be expected to work in the same way for both the parent and the view.
"""
struct SubStateSpaceSet{D, T, V, N, P, S<:SubArray{V,1}} <: AbstractStateSpaceSet{D,T,V,N}
    parent::P
    data::S
    names::N
end
function SubStateSpaceSet(par, data)
    @assert parent(data) === par.data
    P = typeof(par)
    S = typeof(data)
    SV = eltype(P)
    T = eltype(SV)
    D = length(SV)
    V = containertype(par)
    N = eltype(par.names)
    SubStateSpaceSet{D,T,V,N,P,S}(par, data, par.names)
end

function Base.summary(sd::SubStateSpaceSet{D, T}) where {D, T}
    N = length(sd)
    return "$N-element view of $D-dimensional StateSpaceSet{$(T)}"
end

Base.parent(sd::SubStateSpaceSet) = sd.parent
Base.parentindices(sd::SubStateSpaceSet) = parentindices(vec(sd))

"""
    view(d::StateSpaceSet, indices)

Return a view into the parent dataset `d`, as a [`SubStateSpaceSet`](@ref)
that contains the datapoints of `d` referred to by `indices`.
"""
Base.view(d::AbstractStateSpaceSet, i) = SubStateSpaceSet(d, view(vec(d), i))

function Base.view(::AbstractStateSpaceSet, ::Any, ::Any, ::Vararg)
    throw(ArgumentError("StateSpaceSet views only accept indices on one dimension"))
end
