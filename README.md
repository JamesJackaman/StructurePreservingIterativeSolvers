# Structure preserving iterative linear solvers

**This repository is a suppliment to _“Preconditioned Krylov solvers for structure-preserving discretisations” by Jackaman and Maclachlan_, and contains a conservative GMRES implementation in addition to experiments for various constrained linear systems.**

## Installation

### Install the latest version of all components

This repository requires [Firedrake](https://www.firedrakeproject.org/) and [Irksome](https://firedrakeproject.github.io/Irksome). The Firedrake install script can be downloaded [here](https://www.firedrakeproject.org/download.html), and can be installed (with Irksome) into a new virtual envionrment with `python3 firedrake-install --install irksome`. Irksome is only a requirement for the [Linear KdV equation with RK temporal discretisation](#runge-kutta-in-time).

### Install a stable version of all components

This repository is not intended to maintain future compatibility with its dependencies, as this may effect the reproducabilty of results displayed here. To install a stable version of the code we recommend the following.

1. To install a stable version of Firedrake version into a new virtual environment use `python3 firedrake-install 10.5281/zenodo.7414962`. This will install the version of Firedrake used in the associated paper. 
   - The Zenodo URL can be found [here](https://zenodo.org/record/7414962).
   - The DOI is 10.5281/zenodo.7414962.
   - The tag in the [Firedrake repo](https://github.com/firedrakeproject/firedrake) is `Firedrake_20221208.0`. 
3. We recommend a manual installation of [Irksome](https://firedrakeproject.github.io/Irksome) with the git hash `1f5d7d6800a1f03ba1ce4f755fc500e33415966b` checked out.

### First thing to run

Before using this code it is recommended to run `setup.py` inside the virtual environment. This will install the additional dependency [pandas](https://pandas.pydata.org), in addition to creating the subfolder `plots`. By default,  `setup.py` will install the latest versions of pandas. The stable version is  `1.5.0`. 

## What is CGMRES?

CGMRES is a constrained GMRES algorithm, where before termination a list of user-specified constraints are enforced, the core functionality is described in more detail in **[here](docs/solvers.md)**. 

## Application of CGMRES to test problems

We consider a variety of test problems, which are self contained within subfolders. Each test problem contains a subset of the following python scripts which can be called. More details on these functions can be found **[here](docs/experiments.md)**.

**[SingleSolve.py](docs/experiments.md#singlesolve.py)**: _Solves a single step for a given problem for a variety of linear solvers, comparing the error (to a direct solver), residual and deviation in conserved quantity. Results are typically tabulated and plotted (automatically by `visualise.py`)._

**[Evolve.py](docs/experiments.md#evolve.py)**: _Generates a solution over all time steps using a given linear solver. Outputs the global deviation in conserved quantities. Sometimes also outputs errors over time._

**[Error generation](docs/experiments.md#error-generation)**: _This is only implemented for [Runge-Kutta formulation of linear KdV](#runge-kutta-in-time) and is composed of `ErrorGenerator.py` (generates errors in parallel), `subcall.py` (a call to [Evolve.py](docs/experiments.md#evolve.py) using an argparser), and `ErrorPlotter.py` (generates plots by reading data written in `ErrorGenerator.py`)_

**Plotting**: Almost all of the above scripts generate plots. By default, these plots will be saved to the subfolder `plots` and are not shown interactively.

The additional auxiliary modules are also required, we describe them here for completeness. 

**[SelfTitled.py](docs/experiments.md#selftitled.py)**: _The file sharing the name with the subfolder contains the assembly of the linear system and constraints._

**[LinearSolver.py](docs/experiments.md#linearsolver.py)**: _Wrappers for the core solver functions given in `solvers.py`. In practice, these wrappers are only important for CGMRES to incorporate the constraints._

**refd.py**: _An auxiliary file which converts the linear system back to a Firedrake finite element function, in addition to restructuring where appropriate._

**visualise.py**: _Postprocess `SingleSolve.py` by printing a table containing residuals and deviations in constraints. Visualise the table via matplotlib._

## Test problems

The linear systems we solve correspond to linear finite element approximations, for details of the discretisation see [this paper](#structure-preserving-iterative-linear-solvers). Below we describe the PDE, the continuous form of the invariants and the code implemented for a given experiments.

### 1D Linear KdV equation

We approximate the solution of
```math
u_t + u_x + u_{xxx} = 0,
```
which conserves a mass $\int_\Omega u dx$, momentum $\frac12 \int_\Omega u^2 dx$, and energy $\frac12 \int_\Omega u_x^2 - u^2 dx$ over a spatially periodic domain.

#### Second order temporal discretisation

This experiments code can be found in the folder lkdv.

| Executable functions                  | Implemented        |
| ------------------------------------- | ------------------ |
| `SingleSolve.py`                      | :white_check_mark: |
| `Evolve.py`                           | :white_check_mark: |
| `ErrorGenerator.py`/`ErrorPlotter.py` | :x:                |

#### Runge-Kutta in time

These experiments can be found in lkdvRK.

| Executable functions                  | Implemented        |
| ------------------------------------- | ------------------ |
| `SingleSolve.py`                      | :white_check_mark: |
| `Evolve.py`                           | :white_check_mark: |
| `ErrorGenerator.py`/`ErrorPlotter.py` | :white_check_mark: |

### 2D Linear rotating shallow water equations

We approximate the solution of 
```math
{\bf u}_t + f {\bf u}^{\perp} + c^2 \nabla\rho = 0 \\
\rho_t + \nabla \cdot {\bf u} = 0, 
```
where $f$ and $c$ are constants, and
```math
{\bf u}^{\perp} = {\bf e}_3 \times {\bf u}
.
```
This PDE conserves a mass $\int_\Omega \rho \ dx$ and an energy $\frac12 \int_\Omega {\bf u}^2 + c^2 \rho^2 \ dx$ over periodic domains. Code corresponding to this problem can be found in the subfolder `swe`.

| Executable functions                  | Implemented        |
| ------------------------------------- | ------------------ |
| `SingleSolve.py`                      | :white_check_mark: |
| `Evolve.py`                           | :white_check_mark: |
| `ErrorGenerator.py`/`ErrorPlotter.py` | :x:                |

### 2D Heat equation

We approximate the solution of 
```math
u_t - \Delta u = 0
,
```
which preserves mass $\int_\Omega u dx$ and dissipates energy $\frac{d}{dt} \int_\Omega u^2 \ dx = - \int_\Omega \nabla u \cdot \nabla u \ dx $. Here the energy is constrained to match the dissipation rate of the numerical scheme.

| Executable functions                  | Implemented        |
| ------------------------------------- | ------------------ |
| `SingleSolve.py`                      | :white_check_mark: |
| `Evolve.py`                           | :x:                |
| `ErrorGenerator.py`/`ErrorPlotter.py` | :x:                |
