# Recreate figures

Install the Python dependencies in `Figures/requirements-figures.txt`, then run from the repository root:

```text
python Figures/reproduce.py list
python Figures/reproduce.py build bratu regular-transport training-transfer
python Figures/reproduce.py build mesh-refinement
```

The figure catalogue lists the available selectors. For an individual figure:

```text
python Figures/reproduce.py show training-histories
python Figures/reproduce.py build training-histories
```

Builders read the supplied numerical results and write figures to the locations listed by the catalogue. They do not launch new simulations or training. The original field images are supplied as image assets; regenerating those views requires the corresponding visualization workflow.

Figure builders and their inputs are listed in `Figures/catalogue.json`. Training histories, solution-error tables, work counts and model checkpoints are retained in `training/`; benchmark field views and case inputs are under `Figures/`.

The [mesh-refinement study](../training/twophasetransport/analysis/shock_refinement/README.md) also supplies all midpoint/final solution profiles and a runner for new simulations. Its figure compares total error against an analytic shock and the propagated inverse-error contribution relative to discretization error.
