@deprecate dataset_distance set_distance
@deprecate datasets_sets_distances setsofsets_distances
@deprecate setsofsets_distance setsofsets_distances
export AbstractDataset, Dataset
const AbstractDataset = AbstractStateSpaceSet
const Dataset = StateSpaceSet

function statespace_sampler(rng::AbstractRNG = Random.GLOBAL_RNG;
        min_bounds=[], max_bounds=[], method="uniform",
        radius::Number=-1,
        spheredims::Int=0, center=zeros(spheredims),
    )

    @warn "Using `statespace_sampler` with keywords is deprecated. Use a region instead."

    if min_bounds ≠ [] && max_bounds != []
        if method == "uniform"
            region = HRectangle(min_bounds, max_bounds)
            return statespace_sampler(region)
        elseif method == "multgauss"
            error("`multigauss` sampling was poorly documented and perhaps incorrect. It is not supported anymore")
        else
            @error("Unsupported boxregion sampling method")
        end
    elseif radius ≥ 0 && spheredims ≥ 1
        region = HSphere(radius, center)
        return statespace_sampler(region)
    else
        @error("Incorrect keyword specification.")
    end
end
