# CRIS

Code, trained models, benchmark inputs and plotting data for the Coupled Recurrence Inverse System (CRIS) manuscript. CRIS learns a reusable inverse of a local discrete equation and assembles these inverses into a coupled system for global solution.

The examples cover branch selection in Bratu and two-phase transport on one-dimensional, SPE10, Norne and fractured EDFM grids.

## Getting started

- [Install dependencies and build](docs/INSTALLATION.md).
- [Run the transport benchmarks with a pretrained inverse](docs/SIMULATIONS.md).
- [Reproduce the mesh-refinement accuracy study](training/twophasetransport/analysis/shock_refinement/README.md).
- [Train local inverses](docs/TRAINING.md).
- [Recreate the figures from the supplied results](docs/FIGURES.md).

The transport benchmarks can be run without retraining. Replotting the saved results requires Python but does not require compiling the simulator.

## Repository contents

| Directory | Contents |
| --- | --- |
| `src/` | Coupled transport solver, CRIS implementation and numerical utilities |
| `training/` | Local-equation datasets, native trainers, trained checkpoints and experiment scripts |
| `Figures/` | Benchmark case inputs, saved results, field images and figure builders |
| `docs/` | Installation and reproduction instructions |

## License

The project is distributed under the [MIT license](LICENSE). Bundled third-party code retains its own copyright and license notices.
