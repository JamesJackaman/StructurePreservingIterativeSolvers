# Solvers

Here we describe the functions in `solvers.py`. The script is comprised of three functions. The first is an implementation of FGMRES, the second an implementation of CGMRES, and the final a prototypical implementation of CGMRES (where more constraints are enforced each iteration). Throughout, we are solving
```math
A {\bf x} = {\bf b}
,
````
subject to some list of (non)linear constraints `conlist`.

## FGMRES

The FGMRES implementation here is right preconditioned. It has been hand-coded in a consistent way with CGMRES to allow for a fair comparison.

### Input

- `A` and `b`: Corresponding to the linear system being solved.
- `x0`: Initial guess.
- `k`: Maximum number of iterations.
- `tol`: Solver tolerance.
- `pre`: Preconditioner with either a `solve` attribute or which can be directly multiplied with a vector via `@`. 

### Output

- `x`: Solution vector.
- A dictionary containing:
- - The name of the algorithm.
  - The approximation of `x` at every iteration.
  - The residual `r = Ax - b` at every iteration (without the initial residual).

## CGMRES

The inputs and outputs of CGMRES contain those of FGMRES. To avoid unnecessary exposition we shall not repeat these but only list additional contributions.

Here we use GMRES up to some user-specified tolerance `contol * tol`, and below this enforce constraints via a trust region method. If, for any reason, the constraint is not successfully enforced, the algorithm will not terminate (unless the maximum number of iterations `k` has been reached). If the constrained solve fails, which is possible when there is insufficient freedom in the Krylov space and the problem becomes overdetermined, FGMRES will be used and the algorithm will not terminate.

### Additional inputs

- `conlist`: A list containing all constraints on the linear system. Note the linear system *must* naturally preserve these constraints for the algorithm to be efficient and accurate. 

- `contol`: A constant larger than 1 determining when constraints should initially be enforced. Constrained solves are used when the residual is less than `contol * tol`. 

#### Accepted constraint types

The list of constraints `conlist` can take two different forms, and this form for each test problem can be found in [LinearSolver.py](experiments.md#linearsolver.py). In the general case, `conlist` is a list of dictionaries where an arbitrary dictionary `const` contains the constraint `const['func']` and its Jacobian `const['jac']`. General constraints are expected to depend on the initial guess $x_0$ and the preconditioned matrix obtained from Arnoldi $Z$. When optimising for speed, the constraints are input as a class containing the symmeteric matrix $M$, vector $v$ and constant $c$ for constraints of the form

```math
(x_0 + Zy)^T M (x_0 + Zy) + v^T (x_0 + Zy) + c = 0
```

where $y$ is the function we minimise in the residual minimisation step and $c$ is constant.

## Prototypical CGMRES

As with CGMRES, the inputs and outputs are similar to FGMRES so we just introduce any differences here. This algorithm, given by the function `cgmres_p`, is used for testing how difficult it is to enforce constraints and follows the following procedure:

1. For the first iteration, FGMRES is used.
2. For the second iteration, CGMRES is used enforcing the first constraint.
3. For the third iteration, CGMRES is used enforcing the first two constraints.
4. This continues until all constraints are enforced.

As the aim is to measure how difficult it is to enforce constraints, no solver tolerances are defined here and `k` iterations are computed every time. This algorithm will be utilised in [SingleSolve.py](experiments.md#singlesolve.py). 

### Differences in inputs

- No `tol` is specified, the algorithm terminates by hitting the maximum number of iterations.
- `conlist`: A list containing all constraints on the linear system. These constraints are enforced one-by-one following the ordering of the list.
