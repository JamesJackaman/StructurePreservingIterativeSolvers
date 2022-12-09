# Numerical experiments with CGMRES

Here we describe the functionality of python files we use in the examples. 

## SingeSolve.py

### Functionality

Solves a single step for a given problem for prototypical CGMRES, FGMRES and exact linear solvers. The relative error is compared against the exact linear solver and enforcement of the constraints is checked. The auxiliary file `visualise.py` is then typically used to study the deviation in invariants at each iteration and to plot this graphically. 

## Evolve.py

### Functionality

Generates the solution over all time steps using a specfied linear solver. When executing this file two runs will be performed with FGMRES and CGMRES, and then plotted.

### Input

- `N`: Number of time steps

- `M`: Number of spatial elements

- `degree`: spatial polynomial degree

- `T`: End time

- `k`: Maximum number of iterations of linear solver on each step

- `tol`: Solver tolerance

- `contol`: Start enforcing constraints once residual is below `contol * tol`. 

- `solver`: A python function which corresponds to a wrapped linear solver as constructed in [LinearSolver.py](#linearsolver.py).

If using a RK temporal discretisation:

- `tstages`: Number of stages in the GLRK method.

### Output

A python dictionary containing:

- `sol`: An array of the solution as Firedrake functions.

- `time`: Time corresponding to `sol`.

- `err`: An optional output, if an exact solution exists and we are interested in it this is output here.

- A list (per constraint) containing the deviation in a given constraint compared to its initial value. These are denoted as `dm`, `dmo`, and/or `de`. 

## Error generation

Here we refer to three python files, which combined compute and visualise the errors for a variety of spatial and temporal resolutions. This is only implemented for the [Runge-Kutta formulation of linear KdV](../README.md#runge-kutta-in-time). 

### subcall.py

#### Functionality

Call `Evolve.py` with an argparser for variable `degree`. `tstages` and `tol`. Used for parallel calls and to avoid error generation failing if an error is thrown in one instance of `Evolve.py`. 

#### Input

Users can specify in argparser

- `solver`: A string, either `’CGMRES'`, `‘GMRES’` or `‘Exact’`, which will be converted to corresponding function in `LinearSolver.py`.
- `degree`: Spatial degree.
- `tstages`: Number of RK stages.
- `N`: Number of time steps.
- `M`: Number of spatial elements.
- `k`: Maximum number of linear solver iterations per time step.
- `tol`: Solver tolerance.

#### Output

There is no output to this function, however, the solution error is written to file to be read later by `ErrorGenerator.py`.

### ErrorGenerator.py

#### Functionality

Generates errors in parallel with a maximum of `MaxProcesses`  parallel processes through calling `subcall.py`. After all subcalls are completed combines all generated data into the file `tmp/error.pickle`.

Loops over a list of solvers, inside this loop an additional list is iterated over which specifies `degree`, `tstages` and `tol`. This, with minor modifications, may be easily generalised.

### ErrorPlotter.py

#### Functionality

Reads `tmp/error.pickle` and plots error over time for all simulations.

## SelfTitled.py

Here we refer to the python file with the same name as the folder containing it. For example, `lkdv.py` in the folder lkdv. 

### Functionality

Here the linear system and constraints are assembled using Firedrake (and possibly Irksome). In addition, here Firedrake dependent operations are conducted, such as computing the value of any conserved quantities, or computing the error.

### problem class

Here the parameters of the discretisation are defined, in addition to defining the finite element function space and either an exact solution or an initial condition.

#### Input

- `N`: # time steps.
- `M` # spatial elements.
-  `degree`: Spatial finite element degree.
-  `T`: End time.
-   `tstages` _(optional)_ Stages in RK method.

#### Attributes

- Input parameters and other problem dependent parameters (e.g. `mlength` the length of the domain).
- `function_space`: A Firedrake object defining the space of functions we solve over.
- `exact`/`ic`: A function describing either the exact solution or the specified initial condition.

### linforms function

This function builds (on top of the problem class) the linear system, in addition to the vectors and matrices needed to enforce the constraints.

#### Input

- The input argments of the [problem class](#problem-class).
- `t`: The time we are solving for (only important if adding explicit time dependency).
- `zinit`: By default `problem.exact` or `problem.ic` specify the initial condition and `zinit=None`. When specified the initial condition will instead be given by the Firedrake function `zinit`. This is required when evolving over time.

#### Output

- `out`: An output dictionary containing all linear operators, the value of the constraints and the vector form of the initial condition. 
- `prob`: The [problem class](#problem-class).

### compute_invariants function

Takes a solution vector and computes the value of quantities which should be invariant.

#### Input

- `prob`: The problem class
- `uvec`: A solution vector corresponding to a finite element function
- `uold`: _(optional)_ Solution vector at the previous time / initial condition, only required for heat equation as energy dissipates (and depends on previous energy).

#### Output

- A dictionary containing the values of all quantities which should be invariant.

### compute_error function

Computes the error, only implemented for [Runge-Kutta formulation of linear KdV](../README.md#runge-kutta-in-time). 

#### Input

- `params`: The first output argument of [linforms](#linforms-function). 
- `prob`: The problem class.
- `zbig`: The solution vector, which is solved for the RK stages.
- `t`: Current time.

#### Output

- `err`: The $L_2$ error at time `t`. 

### z1calc function

A function specific to [Runge-Kutta temporal discretisations](../README.md#runge-kutta-in-time), which maps the solution at the stage values to the solution at the next time level.

#### Input

- `prob`: The problem class
- `zbig`: The solution vector, given as a stacking of the solution at the stage values.
- `z0`: The solution vector at the previous time step (or initial condition).

#### Output

- `z1`: The solution vector at the next time step.



## LinearSolver.py

### Functionality

Wrapper for the [linear solvers](solvers.md) incorporating problem specific information (constraints) for CGMRES. FGMRES and/or exact linear solvers do not need to be wrapped, this is done so the functions may be called in the same way.

When utilising CGMRES, if the solver tolerance is set to be far below machine precision (`1e-20`) the solver defaults to a [prototypical](solvers.md#prototypical-cgmres) one which enforces constraints one-by-one. If the tolerance is reasonable a more [practical algorithm](solvers.md#cgmres) is used and the constraints are enforced near convergence.

### Input

`dic`: A python dictionary generated in [SelfTitled.py](#SelfTitled.py) containing the linear system and the vectors and matrices needed to form the constraints.

`x0`: An initial guess for the solvers.

`k`: Maximum number of iterations.

`tol`: Solver tolerance.

`contol`: Start enforcing constraints once residual is below `contol * tol`. 

`pre`: Optional preconditioning.

`prob`: A dictionary generated in [SelfTitled.py](#selftitled.py) containing problem dependent objects required for solving the problem. Only an input for [Runge-Kutta temporal discretisations](../README.md#runge-kutta-in-time), as the Butcher tableau and time step  size is needed to form constraints in terms of the solution at the next time level. 

### Output

The same as the output of the linear solver in [solvers.py](solvers.md). The first output is the solution given by the algorithm, while the second is a dictionary containing auxiliary information used in other components of the code. 