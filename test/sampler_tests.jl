using StateSpaceSets
using LinearAlgebra, Random, Statistics, Test

@testset "rectangular box" begin
    @testset "$(method)" for method in ("uniform", "multgauss")
        rng = MersenneTwister(1234)  # Seed random number generator for reproducibility

        # Define rectangular box region
        min_bounds = [-1, -2]
        max_bounds = [1, 2]

        # Generate sampler and isinside functions
        gen, isinside = statespace_sampler(rng; min_bounds=min_bounds, max_bounds=max_bounds, method=method)

        # Test generated points are inside the box region
        for i in 1:250
            x = gen()
            @test all(min_bounds .<= x .<= max_bounds)
            @test isinside(x)
        end
    end
end

@testset "sphere" begin
    for radius in (0.1, 4.0)
        for spheredims in (2, 4)

            rng = MersenneTwister(1234)  # Seed random number generator for reproducibility

            center = fill(rand(), spheredims)

            # Generate sampler and isinside functions
            gen, isinside = statespace_sampler(rng; radius=radius, spheredims=spheredims, center=center)

            # Test generated points are inside the sphere region
            for i in 1:50
                x = gen()
                @test norm(x - center) ≤ radius
                @test isinside(x)
            end

        end
    end
end