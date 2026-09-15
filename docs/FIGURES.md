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

Selectors beginning with `manuscript-` produce the compact layouts used in the
manuscript; the other selectors also include larger views of the same studies.
For example:

```text
python Figures/reproduce.py build manuscript-transport-inputs manuscript-linear-budget manuscript-bratu-3d
```

`python Figures/build_manuscript_compact.py` exports all compact layouts to
`Figures/manuscript_compact/`. Use `--only` followed by the layout names shown
in `--help` to export a subset.

Builders read the supplied numerical results and write figures to the locations listed by the catalogue. To generate new numerical results, follow the [simulation](SIMULATIONS.md) or [training](TRAINING.md) instructions. The field images are supplied as image assets; recreating these views requires exporting the simulation fields in a visualization program such as ParaView.

Figure builders and their inputs are listed in `Figures/catalogue.json`. Training histories, solution-error tables, work counts and model checkpoints are retained in `training/`; benchmark field views and case inputs are under `Figures/`.

Figure manifests are generated alongside the exports. They identify input files
by checksum and record calculated summaries and plot dimensions;
`supplementary_methods_manifest.json`, for example, is output from its builder,
not a configuration file to edit before plotting.

The [mesh-refinement study](../training/twophasetransport/analysis/shock_refinement/README.md) also supplies all midpoint/final solution profiles and a runner for new simulations. Its figure compares total error against an analytic shock and the propagated inverse-error contribution relative to discretization error.
