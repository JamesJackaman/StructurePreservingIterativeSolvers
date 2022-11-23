# Structure preserving iterative linear solvers

**This repository is a suppliment to  _JackamanMaclachlan2022_, and contains a conservative GMRES implementation in addition to experiments for various constrained linear systems.**

## Installation

This repository requires [Firedrake](https://www.firedrakeproject.org/) and [Irksome](https://firedrakeproject.github.io/Irksome). The Firedrake install script can be downloaded [here](https://www.firedrakeproject.org/download.html), and can be installed (with Irksome) into a new virtual envionrment with `python3 firedrake-install --install irksome`. Irksome is only a requirement for the [Linear KdV equation with RK temporal discretisation](#runge-kutta-in-time).

Before using this code it is recommended to run `setup.py` inside the virtual environment. This will install the additional dependencies [pandas](https://pandas.pydata.org) and [pickle](https://docs.python.org/3/library/pickle.html), in addition to creating the subfolder `plots`.

## What is CGMRES?

CGMRES is a constrained GMRES algorithm, where before termination a list of user-specified constraints are enforced, the core functionality is described in more detail in **[here](docs/solvers.md)**. 

## Application of CGMRES to test problems

We consider a variety of test problems, which are self contained within subfolders. Each test problem contains a subset of the following python scripts which can be called. More details on these functions can be found **[here](docs/experiments.md)**.

**[SingleSolve.py](docs/experiments.md#singlesolve.py)**: _Solves a single step for a given problem for a variety of linear solvers, comparing the error (to a direct solver), residual and deviation in conserved quantity. Results are typically tabulated and plotted by `visualise.py`._

**[evolve.py](docs/experiments.md#evolve.py)**: _Generates a solution over all time steps using a given linear solver. Outputs the global deviation in conserved quantities. Sometimes also outputs errors over time._

**[Error generation](docs/experiments.md#error-generation)**: _This is only implemented for [Runge-Kutta formulation of linear KdV](#runge-kutta-in-time) and is composed of `ErrorGenerator.py` (generates errors in parallel), `subcall.py` (a call to [evolve.py](docs/experiments.md#evolve.py) using an argparser), and `ErrorPlotter.py` (generates plots by reading data written in `ErrorGenerator.py`)_

The additional auxiliary modules are also required, we describe them here for completeness. 

**[SelfTitled.py](docs/experiments.md#selftitled.py)**: _The file sharing the name with the subfolder contains the assembly of the linear system and constraints._

**[LinearSolver.py](docs/experiments.md#linearsolver.py)**: _Wrappers for the core solver functions given in `solvers.py`. In practice, these wrappers are only important for CGMRES to incorporate the constraints._

**refd.py**: _An auxiliary file which converts the linear system back to a Firedrake finite element function, in addition to restructuring where appropriate._

**visualise.py**: _Postprocess `SingleSolve.py` by printing a table containing residuals and deviations in constraints. Visualise the table via matplotlib._

## Test problems

The linear systems we solve correspond to linear finite element approximations, for details of the discretisation see _JackamanMaclachlan2022_. Below we describe the PDE, the continuous form of the invariants and the code implemented for a given experiments.

### 1D Linear KdV equation

We approximate the solution of
```math
u_t + u_x + u_{xxx} = 0,
```
which conserves a mass $\int_\Omega u dx$, momentum $\frac12 \int_\Omega u^2 dx$, and energy $\frac12 \int_\Omega u_x^2 - u^2 dx$ over a spatially periodic domain.

#### Second order temporal discretisation

This experiments code can be found in the folder lkdv.

| Executable functions | Implemented        |
| -------------------- | ------------------ |
| `SingleSolve.py`     | :white_check_mark: |
| `evolve.py`          | :white_check_mark: |
| Error generation     | :x:                |

#### Runge-Kutta in time

These experiments can be found in lkdvRK.

| Executable functions | Implemented        |
| -------------------- | ------------------ |
| `SingleSolve.py`     | :white_check_mark: |
| `evolve.py`          | :white_check_mark: |
| Error generation     | :white_check_mark: |

#### 2D Linear rotating shallow water equations

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
This PDE conserves a mass $\int_\Omega \rho \ dx$ and an energy $\frac12 \int_\Omega {\bf u}^2 + c^2 \rho^2 \ dx$ over periodic domains. Code corresponding to this problem can be found in swe.

| Executable functions | Implemented        |
| -------------------- | ------------------ |
| `SingleSolve.py`     | :white_check_mark: |
| `evolve.py`          | :white_check_mark: |
| Error generation     | :x:                |

### 2D Heat equation

We approximate the solution of 
```math
u_t - \Delta u = 0
,
```
which preserves mass $\int_\Omega u dx$ and dissipates energy $\frac{d}{dt} \int_\Omega u^2 \ dx = - \int_\Omega \nabla u \cdot \nabla u \ dx $. Here the energy is constrained to match the dissipation rate of the numerical scheme.

| Executable functions | Implemented        |
| -------------------- | ------------------ |
| `SingleSolve.py`     | :white_check_mark: |
| `evolve.py`          | :x:                |
| Error generation     | :x:                |
