using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "StateSpaceSets.jl" begin
    testfile("ssset_tests.jl")
    testfile("ssset_distance_tests.jl")
    testfile("utils_tests.jl")
    testfile("sampler_tests.jl")
end
