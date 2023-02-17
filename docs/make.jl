import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/apply_style.jl",
    joinpath(@__DIR__, "apply_style.jl")
)
include("apply_style.jl")

using StateSpaceSets, Neighborhood

STATESPACESETS_PAGES = [
    "index.md",
]

makedocs(
    modules = [StateSpaceSets, Neighborhood],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 3,
        ),
    sitename = "StateSpaceSets.jl",
    authors = "George Datseris",
    pages = STATESPACESETS_PAGES,
    doctest = false,
    draft = false,
)

if CI
    deploydocs(
        repo = "github.com/JuliaDynamics/StateSpaceSets.jl.git",
        target = "build",
        push_preview = true
    )
end
