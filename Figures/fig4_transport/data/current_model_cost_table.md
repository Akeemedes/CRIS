# Transport computational cost

CRIS with residual weight lambda=0.01 and paired SRDM results.
All factors are SRDM/CRIS. Equal-weight work sums Jacobian builds, Krylov
iterations and residual evaluations before taking the ratio. Counts include
failed attempts; one Jacobian build is represented by each Newton update.

| Case | Jacobian factor | Krylov factor | Residual factor | Work factor | CRIS (s) | SRDM (s) | Time factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D | 2.01 | 2.01 | 17.4 | 7.18 | 3.73 | 24 | 6.44 |
| Layer 50 | 7.05 | 11.6 | 117 | 29.1 | 3.68 | 110 | 29.8 |
| Layer 75 | 1.47 | 3.28 | 9.03 | 4.17 | 4.31 | 9.69 | 2.25 |
| Layer 84 | 4.69 | 5.81 | 67.1 | 14.8 | 5.53 | 117 | 21.2 |
| Norne | 12 | 24.8 | 216 | 60.4 | 13.1 | 778 | 59.3 |
| SPE10 | 8.39 | 13.6 | 142 | 34.6 | 357 | 2.11e+04 | 59.1 |
| EDFM | 197 | 369 | 3.85e+03 | 980 | 6.89 | 5.45e+03 | 791 |

Times are measured sequential single-process observations including startup,
pressure and I/O. Reference generation is excluded. These are single realizations,
not repeated-timing confidence intervals. The manifest records threads and host.

## Underlying counts (SRDM / CRIS)

| Case | Jacobian builds | Krylov iterations | Residual evaluations |
|---|---:|---:|---:|
| 1D | 1,536 / 765 | 1,536 / 765 | 13,485 / 777 |
| Layer 50 | 1,353 / 192 | 9,319 / 801 | 24,178 / 206 |
| Layer 75 | 389 / 264 | 2,370 / 723 | 2,529 / 280 |
| Layer 84 | 1,476 / 315 | 10,213 / 1,758 | 24,508 / 365 |
| Norne | 2,276 / 189 | 15,590 / 629 | 43,830 / 203 |
| SPE10 | 1,392 / 166 | 9,395 / 693 | 25,043 / 176 |
| EDFM | 51,152 / 260 | 357,471 / 969 | 1,066,573 / 277 |

## Timing sources

`publication_sources.json` identifies the output files and process time for each paired run.
All cases completed; failed attempts are included. Independent references use CRIS time grids.
Producer curves use physical fractional-flow water cuts reconstructed from completion rates.

## Accuracy and timestep accounting

| Case | Final saturation RMSE | CRIS accepted / failed | SRDM accepted / failed |
|---|---:|---:|---:|
| 1D | 0.001157 | 12 / 0 | 12 / 0 |
| Layer 50 | 9.204e-05 | 14 / 0 | 16 / 1 |
| Layer 75 | 4.895e-05 | 16 / 0 | 16 / 0 |
| Layer 84 | 0.0002065 | 50 / 0 | 51 / 1 |
| Norne | 3.089e-05 | 14 / 0 | 16 / 2 |
| SPE10 | 5.568e-05 | 10 / 0 | 11 / 1 |
| EDFM | 2.762e-05 | 17 / 0 | 67 / 49 |