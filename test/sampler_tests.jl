using StateSpaceSets
using LinearAlgebra, Random, Statistics, Test

@testset "rectangular box" begin
    @testset "D=$(D)" for D in (2, 31)
        seed = 1234  # Seed random number generator for reproducibility

        # Define rectangular box region
        min_bounds = rand(D) .- 5
        max_bounds = rand(D) .+ 5

        # Generate sampler and isinside functions
        region = HRectangle(min_bounds, max_bounds)

        gen, isinside = statespace_sampler(region, seed)

        # Test generated points are inside the box region
        for i in 1:250
            x = gen()
            @test all(min_bounds .<= x .<= max_bounds)
            @test isinside(x)
        end
    end
    @test HRectangle(SVector(0,0), SVector(1,1)) isa HRectangle
end

@testset "sphere" begin
    @testset "r = $(r)" for r in (0.1, 4.0)
        @testset "D=$(D)" for D in (2, 31)
            @testset "inside=$(inside)" for inside in (true, false)
                seed = 1234
                center = fill(rand(), D)

                R = inside ? HSphere : HSphereSurface
                region = R(r, center)

                # Generate sampler and isinside functions
                gen, isinside = statespace_sampler(region, seed)

                # Test generated points are inside the sphere region
                for i in 1:50
                    x = gen()
                    if inside
                        @test norm(x - center) < r
                    else
                        @test norm(x - center) ≈ r
                    end
                    @test isinside(x)
                end
            end
        end
    end
end