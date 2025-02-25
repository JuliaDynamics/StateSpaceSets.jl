export set_distance, setsofsets_distances
export Centroid, Hausdorff, StrictlyMinimumDistance

###########################################################################################
# StateSpaceSet distance
###########################################################################################
"""
    set_distance(ssset1, ssset2 [, distance])

Calculate a distance between two `StateSpaceSet`s,
i.e., a distance defined between sets of points, as dictated by `distance`.

Possible `distance` types are:

- [`Centroid`](@ref), which is the default, and 100s of times faster than the rest
- [`Hausdorff`](@ref)
- [`StrictlyMinimumDistance`](@ref)
- Any function `f(A, B)` that returns the distance between two state space sets `A, B`.
"""
set_distance(d1, d2) = set_distance(d1, d2, Centroid())
set_distance(d1, d2, f::Function; kw...) = f(d1, d2)

"""
    Centroid(metric = Euclidean())

A distance that can be used in [`set_distance`](@ref).
The `Centroid` method returns the distance (according to `metric`) between the
[centroids](https://en.wikipedia.org/wiki/Centroid) (a.k.a. centers of mass) of the sets.

`metric` can be any function that takes in two static vectors are returns a positive
definite number to use as a distance (and typically is a `Metric` from Distances.jl).
"""
struct Centroid{M}
    metric::M
end
Centroid() = Centroid(Euclidean())

function set_distance(d1::AbstractStateSpaceSet, d2::AbstractStateSpaceSet, c::Centroid; kw...)
    c1, c2 = centroid(d1), centroid(d2)
    return c.metric(c1, c2)
end
centroid(A::AbstractStateSpaceSet) = sum(A)/length(A)

"""
    Hausdorff(metric = Euclidean())

A distance that can be used in [`set_distance`](@ref).
The [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) is the
greatest of all the distances from a point in one set to the closest point in the other set.
The distance is calculated with the metric given to `Hausdorff` which defaults to Euclidean.

`Hausdorff` is 2x slower than [`StrictlyMinimumDistance`](@ref), however it is
a proper metric in the space of sets of state space sets.

This metric only works for `StateSpaceSet`s whose elements are `SVector`s.

For developers: `set_distance` can take keywords `tree1, tree2` that are the KDTrees
of the first and second sets respectively.
"""
struct Hausdorff{M<:Metric}
    metric::M
end
Hausdorff() = Hausdorff(Euclidean())

function set_distance(d1::AbstractStateSpaceSet, d2, h::Hausdorff;
        # trees given for performance optimizations downstream
        tree1 = KDTree(d1, h.metric),
        tree2 = KDTree(d2, h.metric),
    )
    ε1 = set_distance_tree(d2, tree1, >)
    ε2 = set_distance_tree(d1, tree2, >)
    return max(ε1, ε2)
end

"""
    StrictlyMinimumDistance([brute = false,] [metric = Euclidean(),])

A distance that can be used in [`set_distance`](@ref).
The `StrictlyMinimumDistance` returns the minimum distance of all the distances from a
point in one set to the closest point in the other set.
The distance is calculated with the given metric.

The `brute::Bool` argument switches the computation between a KDTree-based version,
or brute force (i.e., calculation of all distances and picking the smallest one).
Brute force performs better for sets that are either large dimensional or
have a small amount of points. Deciding a cutting point is not trivial,
and is recommended to simply benchmark the [`set_distance`](@ref) function to
make a decision.

If `brute = false` this metric only works for `StateSpaceSet`s whose elements are `SVector`s.

For developers: `set_distance` can take a keyword `tree2` that is the KDTree
of the second set.
"""
struct StrictlyMinimumDistance{M<:Metric}
    brute::Bool
    metric::M
end
StrictlyMinimumDistance() = StrictlyMinimumDistance(false, Euclidean())
StrictlyMinimumDistance(m::Metric) = StrictlyMinimumDistance(false, m)
StrictlyMinimumDistance(brute::Bool) = StrictlyMinimumDistance(brute, Euclidean())

function set_distance(d1, d2::AbstractStateSpaceSet, m::StrictlyMinimumDistance;
        tree1 = nothing, tree2 = KDTree(d2, m.metric)
    )
    if m.brute
        return set_distance_brute(d1, d2, m.metric)
    else
        return set_distance_tree(d1, tree2)
    end
end

# The comparison version exists because when passing `>` it is used in `Hausdorf`
function set_distance_tree(d1, tree::KDTree, comparison = <)
    if comparison === <
        ε = eltype(d1)(Inf)
    elseif comparison === >
        ε = eltype(d1)(-Inf)
    end
    # We use internal source code extracted from NearestNeighbors.jl for max performance
    dist, idx = [ε], [0]
    for p in d1 # iterate over all points of set
        Neighborhood.NearestNeighbors.knn_point!(
            tree, p, false, dist, idx, Neighborhood.NearestNeighbors.always_false
        )
        @inbounds comparison(dist[1], ε) && (ε = dist[1])
    end
    return ε
end

function set_distance_brute(d1, d2::AbstractStateSpaceSet, metric = Euclidean())
    ε = eltype(d2)(Inf)
    for x ∈ d1
        for y ∈ d2
            εnew = metric(x, y)
            εnew < ε && (ε = εnew)
        end
    end
    return ε
end

###########################################################################################
# Sets of datasets distance
###########################################################################################
"""
    setsofsets_distances(a₊, a₋ [, distance]) → distances

Calculate distances between sets of `StateSpaceSet`s. Here  `a₊, a₋` are containers of
`StateSpaceSet`s, and the returned distances are dictionaries of distances.
Specifically, `distances[i][j]` is the distance of the set in
the `i` key of `a₊` to the `j` key of `a₋`. Distances from `a₋` to
`a₊` are not computed at all, assumming symmetry in the distance function.

The `distance` can be anything valid for [`set_distance`](@ref).

Containers `a₊, a₋` can be empty but they must be concretely typed.
"""
function setsofsets_distances(a₊, a₋, method = Centroid())
    ids₊, ids₋ = keys(a₊), keys(a₋)
    numbertype = a -> eltype(eltype(valtype(a))) # numeric type = type of distance
    T = promote_type(numbertype(a₊), numbertype(a₋))
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), T}}()
    _setsofsets_distances!(distances, a₊, a₋, method)
end

function _setsofsets_distances!(distances, a₊, a₋, c::Centroid)
    centroids₋ = Dict(k => centroid(v) for (k, v) in pairs(a₋))
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        centroid_A = centroid(A)
        for m in keys(a₋)
            distances[k][m] = c.metric(centroid_A, centroids₋[m])
        end
    end
    return distances
end

function _setsofsets_distances!(distances, a₊, a₋, method::Hausdorff)
    metric = method.metric
    trees₊ = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₊))
    trees₋ = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₋))
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        tree1 = trees₊[k]
        for (m, tree2) in trees₋
            # Internal method of `set_distance` for Hausdorff
            d = set_distance(A, a₋[m], method; tree1, tree2)
            distances[k][m] = d
        end
    end
    return distances
end

function _setsofsets_distances!(distances, a₊, a₋, f::Function)
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        for (m, B) in pairs(a₋)
            distances[k][m] = f(A, B)
        end
    end
    return distances
end

function _setsofsets_distances!(distances, a₊, a₋, method::StrictlyMinimumDistance)
    @assert keytype(a₊) == keytype(a₋)
    if method.brute == false
        search_trees = Dict(m => KDTree(vec(att), method.metric) for (m, att) in pairs(a₋))
    end
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        for (m, B) in pairs(a₋)
            d = set_distance(A, B, method; tree2 = search_trees[m])
            distances[k][m] = d
        end
    end
    return distances
end
