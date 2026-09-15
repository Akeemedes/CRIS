# Training local inverses

Training uses labelled scalar roots and an optional residual penalty. The experiment definitions contain the input transformations, constitutive parameters, architecture and loss normalization. The packed datasets and reported checkpoints are included.

```text
python training/experiments.py list
python training/experiments.py show transport.regular.training
```

Three training problems are available:

| Experiment | Local equation |
| --- | --- |
| `transport.regular.training` | Fractional flow with endpoint exponents 2 and 2 |
| `transport.degenerate.training` | Fractional flow with endpoint exponents 2 and 0.2 |
| `bratu.lower_branch.training` | Selected lower branch of the Bratu cell equation |

## Run training

Create a `runs` directory, then choose a new output directory for each run. For example, on Windows:

```text
python training/experiments.py train transport.regular.training --weight 0.01 --rows 62400 --executable build/transport_training/Release/twophasetransport_lm_trainer.exe --run-dir runs/transport_lambda001 --threads 16
```

Use `--weight 0` for root-value fitting alone. Positive weights add the equation-residual penalty. Use `--runtime-dir` as needed for native libraries. On Linux, use `build/transport_training/twophasetransport_lm_trainer` as the executable path.

For Bratu, select `bratu.lower_branch.training` and the `bratu_lm_trainer` executable. For the singular-endpoint law, select `transport.degenerate.training`; supported transport sample counts include 62,400, 125,000, 250,000, 500,000 and 1,000,000.

Run directories contain checkpoints and optimizer histories. `--continue-from` starts a new training stage from the best saved weights on a larger dataset; `--resume-from` restores the saved optimizer state. See `python training/experiments.py train --help` for their required arguments and training limits.

## Supplied models and data

Transport datasets are in `training/twophasetransport/datasets/`, and Bratu datasets are in `training/bratu/datasets/lower_branch/`. Their manifests describe the sampling and normalization.

To generate the four Bratu label datasets with the supplied sampling settings:

```text
python training/bratu/generate_data.py --output runs/bratu_labels
```

Choose a new output directory; the generator leaves existing datasets untouched.
Dataset paths in the generated manifest are relative to that manifest's directory.

The pretrained transport inverse used by the seven-case benchmark launcher is:

```text
training/twophasetransport/analysis/regular_v3/packages/residual_lambda_0p01_final.pt
```

Other reported weights and training histories are under the corresponding `runs/` and `analysis/` directories. `twophasetransport_checkpoint_exporter` converts a native transport checkpoint into the TorchScript package consumed by the simulator; invoke it with `--help` for the export parameters.
