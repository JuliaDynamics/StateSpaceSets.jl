using Test, StateSpaceSets

@testset "Dataset Distances" begin
    @testset "metric" begin
        d1 = float.(range(1, 10; length = 11))
        d2 = d1 .+ 10
        @test set_distance(Dataset(d1), Dataset(d2), StrictlyMinimumDistance()) == 1.0
        @test set_distance(Dataset(d1), Dataset(d2), StrictlyMinimumDistance(true)) == 1.0
        d1 = Dataset([SVector(0.0, 1)])
        d2 = Dataset([SVector(1.0, 2)])
        @test set_distance(d1, d2, StrictlyMinimumDistance(Chebyshev())) == 1.0
    end

    @testset "Hausdorff" begin
        d1 = float.(range(1, 10; length = 11))
        d2 = d1 .+ 10
        @test set_distance(Dataset(d1), Dataset(d2), Hausdorff()) == 10.0
        d1 = Dataset([SVector(0.0, 1)])
        d2 = Dataset([SVector(1.0, 2), SVector(1.0, 3)])
        @test set_distance(d1, d2, Hausdorff(Chebyshev())) == 2.0
    end

    @testset "Centroid" begin
        # centroid is 0, 0.5
        d1 = Dataset([SVector(0.0, 0.0), SVector(0.0, 1.0)])
        # centroid is 1, 0.5
        d2 = Dataset([SVector(1.0, 0.0), SVector(1.0, 1.0)])
        @test set_distance(d1, d2, Centroid()) == 1.0
        @test set_distance(d1, d2, Centroid(Chebyshev())) == 1.0
        # And also with custom function
        custom_f(x, y) = abs(x[2] - y[2])
        @test set_distance(d1, d2, Centroid(custom_f)) == 0.0
    end

end

@testset "Sets of Dataset Distances" begin
    d1 = range(1, 10; length = 11)
    d2 = d1 .+ 10
    d3 = d2 .+ 10
    set1 = Dataset.([d1, d2, d3])
    r = range(1, 10; length = 11)
    D = 6
    makedata = x -> (y = zeros(D); y[1] = x; y)
    d1 = Dataset([makedata(x) for x in r])
    d2 = Dataset([makedata(x+10) for x in r])
    d3 = Dataset([makedata(x+20) for x in r])
    set2 = Dataset.([d1, d2, d3])

    @testset "$method" for method in (StrictlyMinimumDistance(Euclidean()), Hausdorff())
        for set in (set1, set2)

            offset = method isa Hausdorff ? 9 : 0
            dsds = datasets_sets_distances(set, set, method)
            for i in 1:3
                @test dsds[i][i] == 0
            end
            @test dsds[2][1] == dsds[2][3] == 1 + offset
            @test dsds[3][1] == dsds[1][3] == 11 + offset
        end
    end

    @testset "works with dicts" begin
        setx = Dict("x" => d1)
        sety = Dict("y" => d1)
        dsds = datasets_sets_distances(setx, sety)
        @test dsds["x"]["y"] == 0
    end

    @testset "user function" begin
        d4  = d3[1:3]
        set3 = Dataset.([d1, d2, d4])
        f = (d1, d2) -> abs(length(d1) - length(d2))
        dsds = datasets_sets_distances(set3, set3, f)
        @test dsds[2][3] == dsds[3][2] == 8
        @test dsds[2][1] == dsds[1][2] == 0
    end

    @testset "centroid" begin
        d1 = Dataset([SVector(0.0, 0.0), SVector(0.0, 1.0)])
        d2 = Dataset([SVector(1.0, 0.0), SVector(1.0, 1.0)])
        set1 = [d1, d2]
        dsds = datasets_sets_distances(set1, set1, Centroid())
        @test dsds[1][1] == 0
        @test dsds[2][1] == 1
        @test dsds[1][2] == 1
    end
end
