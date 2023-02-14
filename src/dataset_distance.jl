export set_distance, setsofsets_distance
export Centroid, Hausdorff, StrictlyMinimumDistance

###########################################################################################
# Dataset distance
###########################################################################################
"""
    set_distance(dataset1, dataset2 [, method])

Calculate a distance between two `AbstractDatasets`,
i.e., a distance defined between sets of points, as dictated by `method`.

The possible `methods` are:
- [`Centroid`](@ref), which is the default, and 100s of times faster than the rest
- [`Hausdorff`](@ref)
- [`StrictlyMinimumDistance`](@ref)
"""
set_distance(d1, d2) = set_distance(d1, d2, Centroid())

"""
    Centroid(metric = Euclidean())

A distance that can be used in [`set_distance`](@ref).
The `Centroid` method returns the distance (according to `metric`) between the
[centroids](https://en.wikipedia.org/wiki/Centroid) (a.k.a. center of mass) of the sets.

Besides giving as `metric` an instance from Distances.jl, you can give any function that
takes in two static vectors are returns a positive definite number to use as a distance.
"""
struct Centroid{M}
    metric::M
end
Centroid() = Centroid(Euclidean())

function set_distance(d1::AbstractDataset, d2::AbstractDataset, c::Centroid)
    c1, c2 = centroid(d1), centroid(d2)
    return c.metric(c1, c2)
end
centroid(A::AbstractDataset) = sum(A)/length(A)

"""
    Hausdorff(metric = Euclidean())

A distance that can be used in [`set_distance`](@ref).
The [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) is the
greatest of all the distances from a point in one set to the closest point in the other set.
The distance is calculated with the metric given to `Hausdorff` which defaults to Euclidean.

`Hausdorff` is 2x slower than [`StrictlyMinimumDistance`](@ref), however it is
a proper metric in the space of sets of datasets.
"""
struct Hausdorff{M<:Metric}
    metric::M
end
Hausdorff() = Hausdorff(Euclidean())

function set_distance(d1::AbstractDataset, d2, h::Hausdorff,
        # trees given for a natural way to call this function in `setsofsets_distance`
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
Brute force performs better for datasets that are either large dimensional or
have a small amount of points. Deciding a cutting point is not trivial,
and is recommended to simply benchmark the [`set_distance`](@ref) function to
make a decision.
"""
struct StrictlyMinimumDistance{M<:Metric}
    brute::Bool
    metric::M
end
StrictlyMinimumDistance() = StrictlyMinimumDistance(false, Euclidean())
StrictlyMinimumDistance(m::Metric) = StrictlyMinimumDistance(false, m)
StrictlyMinimumDistance(brute::Bool) = StrictlyMinimumDistance(brute, Euclidean())

function set_distance(d1, d2::AbstractDataset, m::StrictlyMinimumDistance)
    if m.brute
        return set_distance_brute(d1, d2, m.metric)
    else
        tree = KDTree(d2, m.metric)
        return set_distance_tree(d1, tree)
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

function set_distance_brute(d1, d2::AbstractDataset, metric = Euclidean())
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
    setsofsets_distance(a₊, a₋ [, method]) → distances

Calculate distances between sets of `Dataset`s. Here  `a₊, a₋` are containers of
`Dataset`s, and the returned distances are dictionaries of
of distances. Specifically, `distances[i][j]` is the distance of the set in
the `i` key of `a₊` to the `j` key of `a₋`. Notice that distances from `a₋` to
`a₊` are not computed at all (assumming symmetry in the distance function).

The `method` can be as in [`set_distance`](@ref).
However, `method` can also be any arbitrary user function that takes as input
two datasets and returns any positive-definite number as their "distance".
"""
function setsofsets_distance(a₊, a₋, method = Centroid())
    (isempty(a₊) || isempty(a₋)) && error("The set containers must be non-empty.")
    ids₊, ids₋ = keys(a₊), keys(a₋)
    gettype = a -> eltype(first(values(a)))
    T = promote_type(gettype(a₊), gettype(a₋))
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), T}}()
    _datasets_sets_distances!(distances, a₊, a₋, method)
end

function _datasets_sets_distances!(distances, a₊, a₋, c::Centroid)
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

function _datasets_sets_distances!(distances, a₊, a₋, method::Hausdorff)
    @assert keytype(a₊) == keytype(a₋)
    metric = method.metric
    trees₊ = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₊))
    trees₋ = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₋))
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        tree1 = trees₊[k]
        for (m, tree2) in trees₋
            # Internal method of `set_distance` for Hausdorff
            d = set_distance(A, a₋[m], method, tree1, tree2)
            distances[k][m] = d
        end
    end
    return distances
end

function _datasets_sets_distances!(distances, a₊, a₋, f::Function)
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        for (m, B) in pairs(a₋)
            distances[k][m] = f(A, B)
        end
    end
    return distances
end

function _datasets_sets_distances!(distances, a₊, a₋, method::StrictlyMinimumDistance)
    @assert keytype(a₊) == keytype(a₋)
    if method.brute == false
        search_trees = Dict(m => KDTree(vec(att), method.metric) for (m, att) in pairs(a₋))
    end
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        for (m, B) in pairs(a₋)
            if method.brute == false
                # Internal method of `set_distance` for non-brute way
                d = set_distance_tree(A, search_trees[m])
            else
                d = set_distance_brute(A, B, method.metric)
            end
            distances[k][m] = d
        end
    end
    return distances
end
