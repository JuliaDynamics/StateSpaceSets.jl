# StateSpaceSets.jl

[![docsdev](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/statespacesets/dev/)
[![docsstable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/statespacesets/stable/)
[![](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--91032--7-purple)](https://link.springer.com/book/10.1007/978-3-030-91032-7)
[![CI](https://github.com/JuliaDynamics/StateSpaceSets.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/StateSpaceSets.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaDynamics/StateSpaceSets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDynamics/StateSpaceSets.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/StateSpaceSets)](https://pkgs.genieframework.com?packages=StateSpaceSets)

A Julia package that provides functionality for state space sets.
These are collections of points of fixed, and known by type, size (called dimension).
It is used by many other packages in the JuliaDynamics organization.
The main export of `StateSpaceSets` is the concrete type `StateSpaceSet`.
The package also provides functionality for distances, neighbor searches,
sampling, and normalization.

To install it you may run `import Pkg; Pkg.add("StateSpaceSets")`,
however, there is no real reason to install this package directly
as it is re-exported by all downstream packages that use it.