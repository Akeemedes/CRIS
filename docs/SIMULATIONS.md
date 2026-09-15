# Transport simulations

The paired benchmark launcher uses the supplied transport inverse on seven cases: 1D, three SPE10 layers, three-dimensional SPE10, Norne and fractured EDFM. Each pair uses the same spatial discretization, pressure solution, time-step controls and convergence tolerances. SRDM uses line search; CRIS uses full Newton updates.

## Run the seven paired benchmarks

After [building the simulator](INSTALLATION.md), choose a new output directory. The following Windows commands prepare and run the cases:

```text
python training/twophasetransport/scripts/publication_benchmarks.py stage --output runs/transport --solver build/transport/Release/tp_transport.exe --threads 1 --linmax 7
python training/twophasetransport/scripts/publication_benchmarks.py run --output runs/transport --solver build/transport/Release/tp_transport.exe --threads 1 --linmax 7
```

On Linux, replace the executable path with `build/transport/tp_transport`. Add `--dll-dir` for runtime libraries if needed. The larger cases can take hours. Repeating the run command resumes the campaign without rerunning completed, unchanged cases.

Each case has `CRIS` and `SRDM` subdirectories containing `sim.txt` and solver outputs. The configuration files define the time-step policy and all numerical tolerances. The solver reports record nonlinear iterations, linear iterations, residual evaluations and timings.

## Compare solutions and plot results

```text
python training/twophasetransport/scripts/publication_benchmarks.py references --output runs/transport --solver build/transport/Release/tp_transport.exe
python training/twophasetransport/scripts/publication_benchmarks.py publish --output runs/transport --solver build/transport/Release/tp_transport.exe
```

Here `publish` generates local plotting files; it does not upload anything. Independent reference solutions evaluate error against the original discrete equations. The supplied manuscript results can also be [plotted directly](FIGURES.md).

## Other experiments

The [shock-refinement study](../training/twophasetransport/analysis/shock_refinement/README.md) compares SRDM and CRIS with analytic cell averages and independent discrete solutions on 19 mesh/timestep pairs. It includes a standalone simulation runner, midpoint/final solution profiles and error tables. Replot the saved results with `python Figures/reproduce.py build mesh-refinement`, or rerun the paired simulations with:

```text
python training/twophasetransport/scripts/shock_refinement.py --solver build/transport/Release/tp_transport.exe --output runs/shock_refinement
```

Saved results and selected model weights for the fixed-point, checkpoint-accuracy and singular-endpoint studies are under `training/twophasetransport/analysis/`. The figure catalogue provides commands for plotting these results.

Bratu branch and initialization studies are implemented under `Figures/fig3_bratu/` and `training/bratu/`. These compare learned inverses with the analytic local inverse.
