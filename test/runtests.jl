using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "StateSpaceSets.jl" begin
    testfile("dataset_tests.jl")
    testfile("dataset_distance_tests.jl")
    testfile("utils_tests.jl")
    testfile("sampler_tests.jl")
end
