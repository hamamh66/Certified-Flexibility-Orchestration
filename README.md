# Certified Flexibility Orchestration (CFO)

## Reliability-Aware Grid Flexibility for Buildings with Electric Vehicles

This repository contains the reference implementation accompanying the
paper:

**Certified Flexibility Orchestration for Grid-Responsive Buildings with
Electric Vehicles: Selective Commitment and the Reliability--Mobility
Trade-off**

The framework introduces **Certified Flexibility Orchestration (CFO)**,
a reliability-aware decision layer that determines **not only how to
dispatch flexibility, but also whether a flexibility commitment should
be made**.

## Key Features

-   Certified Flexibility Envelopes (CFEs)
-   Conformal calibration for uncertainty-aware flexibility
    certification
-   Unified Variational Intelligence Framework (UVIF) decision engine
-   Selective commitment policy (Act / De-rate / Transfer / Abstain)
-   Building--EV integrated energy management
-   Reliability-aware flexibility evaluation
-   Reproducible experimental framework

## Repository Structure

``` text
├── notebooks/          Experimental notebooks
├── src/                Core implementation
├── data/               Example datasets
├── figures/            Figures used in the manuscript
├── results/            Generated experimental results
├── configs/            Simulation configurations
└── docs/               Additional documentation
```

## Implemented Controllers

-   Uncontrolled baseline
-   Time-of-Use control
-   Carbon-aware control
-   Deterministic MPC
-   Robust MPC
-   Certified Flexibility Orchestration (CFO)

## Main Contributions

-   Reliability-aware flexibility certification
-   Certified Flexibility Envelopes
-   Unified variational optimization
-   Selective commitment for trustworthy grid services
-   Reliability--mobility trade-off analysis
-   Fully reproducible research workflow

## Citation

If you use this repository, please cite the accompanying paper.

## License

This project will be released under the MIT License.

## Contact

Please open a GitHub Issue for questions, suggestions, or bug reports.
