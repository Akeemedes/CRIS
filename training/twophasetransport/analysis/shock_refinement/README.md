# Accuracy under refinement

This study compares SRDM and CRIS with an analytic single-shock solution while reusing the supplied transport inverse without retraining. It contains 19 paired meshes across four timestep-to-mesh ratios, with solutions at the midpoint and final horizon.

## Numerical problem

The conservation law is `U_t + f(U)_x = 0`, where `f(U) = U^2 / (U^2 + 0.4(1-U)^2)`. The domain is `[0,1000]`, velocity is one, and gravity and dispersion are zero. Initially, saturation is 0.25 to the left of 250 and 0.05 to the right. The left inflow is fixed at 0.25; the right boundary is advective outflow. The final horizon is 200, with an additional output at 100.

The flux is strictly convex between the two initial states, giving one entropy-admissible shock with speed 1.0525686262783327. Analytic comparisons use exact cell averages, including the cell containing the moving discontinuity. The initial discontinuity lies on a face for every mesh.

The discretization is first-order implicit upwind finite volume with backward Euler. Within each cohort, `dt = beta*h`, where `h = 1000/N` and velocity is one. Refinement therefore decreases both `h` and `dt`; it is not a spatial-only refinement at fixed timestep.

| `beta = dt/h` | Cell counts |
| ---: | --- |
| 1 | 100, 200, 400, 800, 1,600, 3,200 |
| 10 | 2,000, 4,000, 8,000, 10,000, 16,000 |
| 100 | 8,000, 16,000, 20,000, 32,000 |
| 1000 | 20,000, 40,000, 50,000, 80,000 |

Both formulations use nonlinear update and residual tolerances of `1e-10`, seven Krylov iterations per linear solve, and a refreshed ILU preconditioner. SRDM uses line search; CRIS uses full Newton updates. The maximum Newton updates per step are 100 for `beta=1` and 1000 for the larger-step cohorts, identically for both formulations. Every supplied simulation completed its prescribed common time grid without a rejected step.

## Error definitions

Let `U_phi` denote the independently solved original discrete equations, `U_theta` the learned CRIS solution, and `I_h u(T)` the analytic cell averages. The spatially normalized discrete L2 norm is RMSE, `sqrt(mean(error^2))`. At the final horizon:

- `E_disc = ||U_phi - I_h u(T)||`: combined space-time discretization error.
- `E_inv = ||U_theta - U_phi||`: propagated error from the learned inverse.
- `E_tot = ||U_theta - I_h u(T)||`: total CRIS solution error.

The exact discrete reference is computed independently by scalar Brent roots in causal cell order. This is an accuracy reference, not the SRDM computational baseline. SRDM agrees with it to at most `2.981e-14` RMSE across the final-time pairs. The maximum scalar reference residual is below `1e-12`. For `beta=1000`, the scalar root tolerances are tightened because the large local derivative amplifies root-location error.

The sampled `E_inv/E_disc = 1` crossover lies between 1,600 and 3,200 cells for `beta=1`, between 8,000 and 10,000 for `beta=10`, between 20,000 and 32,000 for `beta=100`, and between 40,000 and 50,000 for `beta=1000`. At 20,000 cells and `beta=100`, the ratio is already 0.983. These brackets describe this fixed inverse and physical problem, not a universal mesh limit.

Larger timesteps change the local inverse inputs and reduce the number of time updates. They also increase temporal discretization error. A later crossover does not by itself imply a smaller total error. The main figure therefore shows both total solution error and the ratio of the two contributions. Component norms need not add to the norm of the total error. This finite-resolution study complements the approximation analysis; it does not certify a uniform inverse-error bound or asymptotic convergence with a fixed network.

## Recreate the figure

From the repository root, after installing `Figures/requirements-figures.txt`:

```text
python Figures/reproduce.py build mesh-refinement
```

The builder writes a vector PDF, SVG and PNG. It can also be invoked directly:

```text
python Figures/build_shock_refinement.py --formats pdf svg png
```

The first panel compares CRIS and SRDM total RMSE against the analytic solution. The second shows `E_inv/E_disc`. All 19 mesh pairs are retained, including points beyond the crossover.

## Rerun the simulations

After [building the transport simulator](../../../../docs/INSTALLATION.md), run from the repository root:

```text
python training/twophasetransport/scripts/shock_refinement.py --solver build/transport/Release/tp_transport.exe --output runs/shock_refinement
```

This runs all 19 published pairs. On Linux, use `build/transport/tp_transport`. The runner uses the supplied publication checkpoint by default; `--model` selects another checkpoint. Add repeatable `--runtime-dir` arguments for LibTorch, MKL or OpenMP libraries if they are not already discoverable. `--threads` defaults to one. `--timeout` is the time allowance in seconds for each native solver invocation, defaulting to 900.

A shorter run, or a targeted extension, can select one cohort and explicit aligned meshes:

```text
python training/twophasetransport/scripts/shock_refinement.py --solver build/transport/Release/tp_transport.exe --output runs/shock_refinement_small --beta 10 --meshes 100 2000
```

To reproduce the supplied `beta=1000` cohort alone, use `--beta 1000` without `--meshes`. The initial shock and both report times must align: `N/4` and `N/(10*beta)` must be integers. Runs generate the initial conditions and all case configurations automatically. No pressure or external mesh input is needed for this one-dimensional problem.

The independent reference is computed after each native pair. The finest pairs can take several minutes because the scalar reference performs a root solve for each cell and time step. Completed native runs are reused only when their recorded executable, model, configuration, runner and thread-count identities match. Existing incompatible outputs are preserved; choose a new output directory when changing these inputs.

To plot a new campaign:

```text
python Figures/build_shock_refinement.py --data runs/shock_refinement/errors.csv --output runs/shock_refinement/figure --formats pdf svg png
```

## Supplied data

- `errors.csv`: midpoint and final L1, RMSE (`L2`) and maximum-norm errors for both methods, including discretization and inverse-error contributions.
- `work.csv`: accepted-step counts, Newton/Krylov/residual-evaluation counts and recorded wall times. These runs assess accuracy; the timings are not a comparative performance benchmark.
- `profiles/`: compressed NumPy arrays for every pair, with `x`, `SRDM_100`, `CRIS_100`, `analytic_100`, `discrete_100`, and the corresponding `_200` arrays.
- `provenance.json`: physical and numerical settings, checkpoint/executable identities for the supplied results, and data checksums.

The checkpoint is `training/twophasetransport/analysis/regular_v3/packages/residual_lambda_0p01_final.pt`, SHA256 `ffd1166d3381ea4230cf264609182fddd186f5dc63507e3096f6fa0239176b55`. A locally compiled executable has its own hash; the runner records that identity for the new campaign. Small floating-point differences between builds and platforms are expected.
