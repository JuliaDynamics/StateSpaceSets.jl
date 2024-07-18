# Numerical Data

```@docs
StateSpaceSets
```

!!! info "Timeseries and datasets"
    The word "timeseries" can be confusing, because it can mean a univariate (also called scalar or one-dimensional) timeseries or a multivariate (also called multi-dimensional) timeseries.
    To resolve this confusion, in **DynamicalSystems.jl** we have the following convention: **"timeseries"** is always univariate! it refers to a one-dimensional vector of numbers, which exists with respect to some other one-dimensional vector of numbers that corresponds to a time vector.
    On the other hand, we use the word **"state space set"** to refer to a *multi-dimensional* timeseries, which is of course simply a group/set of one-dimensional timeseries represented as a [`StateSpaceSet`](@ref).

## StateSpaceSet

Trajectories, and in general sets in state space, are represented by a structure called `StateSpaceSet` in **DynamicalSystems.jl**
(while timeseries are always standard Julia `Vector`s).
It is recommended to always [`standardize`](@ref) datasets.

```@docs
StateSpaceSet
```

In essence a `StateSpaceSet` is simply a wrapper for a `Vector` of `SVector`s.
However, it is visually represented as a matrix, similarly to how numerical data would be printed on a spreadsheet (with time being the *column* direction).
It also offers a lot more functionality than just pretty-printing.
Besides the examples in the documentation string, you can e.g. iterate over data points
```julia
using DynamicalSystems
hen = Systems.henon()
data = trajectory(hen, 10000) # this returns a dataset
for point in data
    # stuff
end
```

Most functions from **DynamicalSystems.jl** that manipulate and use multidimensional data are expecting a `StateSpaceSet`.
This allows us to define efficient methods that coordinate well with each other, like e.g. [`embed`](@ref).

## StateSpaceSet Functions
```@docs
minima
maxima
minmaxima
columns
```


## Basic statistics

```@docs
standardize
cor
cov
mean_and_cov
```

## StateSpaceSet distances
### Two datasets
```@docs
set_distance
Hausdorff
Centroid
StrictlyMinimumDistance
```
### Sets of datasets
```@docs
setsofsets_distances
```


## StateSpaceSet I/O
Input/output functionality for an `AbstractStateSpaceSet` is already achieved using base Julia, specifically `writedlm` and `readdlm`.
To write and read a dataset, simply do:

```julia
using DelimitedFiles

data = StateSpaceSet(rand(1000, 2))

# I will write and read using delimiter ','
writedlm("data.txt", data, ',')

# Don't forget to convert the matrix to a StateSpaceSet when reading
data = StateSpaceSet(readdlm("data.txt", ',', Float64))
```

## Neighborhoods
Neighborhoods refer to the common act of finding points in a dataset that are nearby a given point (which typically belongs in the dataset).
**DynamicalSystems.jl** bases this interface on [Neighborhood.jl](https://julianeighbors.github.io/Neighborhood.jl/dev/).
You can go to its documentation if you are interested in finding neighbors in a dataset for e.g. a custom algorithm implementation.

For **DynamicalSystems.jl**, what is relevant are the two types of neighborhoods that exist:
```@docs
NeighborNumber
WithinRange
```

## Samplers
```@docs
statespace_sampler
HSphere
HSphereSurface
HRectangle
```
