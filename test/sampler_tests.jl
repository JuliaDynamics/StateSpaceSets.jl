using LinearAlgebra, Random, Statistics, Test

println("\nTesting sampler...")

@testset "statespace_sampler" begin

# Test statespace_sampler with rectangular box method
function test_statespace_sampler_rectangular_box()
    rng = MersenneTwister(1234)  # Seed random number generator for reproducibility

    # Define rectangular box region
    min_bounds = [-1, -2]
    max_bounds = [1, 2]
    method = "uniform"

    # Generate sampler and isinside functions
    gen, isinside = statespace_sampler(rng; min_bounds=min_bounds, max_bounds=max_bounds, method=method)

    # Test generated points are inside the box region
    for i in 1:250
        x = gen()
        @test all(min_bounds .<= x .<= max_bounds)
        @test isinside(x)
    end
end

# Test statespace_sampler with multivariate Gaussian box method
function test_statespace_sampler_multivariate_gaussian_box()
    rng = MersenneTwister(1234)  # Seed random number generator for reproducibility

    # Define rectangular box region
    min_bounds = [-1, -2]
    max_bounds = [1, 2]
    method = "multgauss"

    # Generate sampler and isinside functions
    gen, isinside = statespace_sampler(rng; min_bounds=min_bounds, max_bounds=max_bounds, method=method)

    # Test generated points are inside the box region
    for i in 1:250
        x = gen()
        @test all(min_bounds .<= x .<= max_bounds)
        @test isinside(x)
    end
end

# Test statespace_sampler with sphere method
function test_statespace_sampler_sphere_3D()
    rng = MersenneTwister(1234)  # Seed random number generator for reproducibility

    # Define sphere region
    radius = 2
    spheredims = 3
    center = [1, 2, 3]

    # Generate sampler and isinside functions
    gen, isinside = statespace_sampler(rng; radius=radius, spheredims=spheredims, center=center)

    # Test generated points are inside the sphere region
    for i in 1:250
        x = gen()
        @test norm(x - center) <= radius
        @test isinside(x)
    end
end

# Test statespace_sampler with sphere method
function test_statespace_sampler_sphere_4D()
    rng = MersenneTwister(1234)  # Seed random number generator for reproducibility

    # Define sphere region
    radius = 2
    spheredims = 4
    center = [1, 2, 3, 7]

    # Generate sampler and isinside functions
    gen, isinside = statespace_sampler(rng; radius=radius, spheredims=spheredims, center=center)

    # Test generated points are inside the sphere region
    for i in 1:250
        x = gen()
        @test norm(x - center) <= radius
        @test isinside(x)
    end
end


# Run tests
    test_statespace_sampler_rectangular_box()
    test_statespace_sampler_multivariate_gaussian_box()
    test_statespace_sampler_sphere_3D()
    test_statespace_sampler_sphere_4D()
end