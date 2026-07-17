# NI-PINN

NI-PINN couples tumor growth, activated CD8 T cells, macrophages, myeloid-derived suppressor cells, norepinephrine, acetylcholine, and drug transport in a physics-informed neural surrogate. A nested fast–medium–slow optimizer resolves neurotransmitter, immune, and tumor timescales. The learned fields supply therapy-aware state variables to a PPO controller for simulated magnetic microrobot navigation in hepatic vasculature.

## Scope

The package contains the coupled equations, automatic-differentiation residuals, neuro-immune pathway loss, synthetic trajectory generator, TCGA-LIHC preprocessing, vascular dynamics, multi-rate training, PPO optimization, statistical metrics, reported-result registry, and experiment catalog. It does not contain clinical decision support and has not been validated for patient care or physical microrobot actuation.

## Environment

The reference environment uses Python 3.10.13, PyTorch 2.1.2, CUDA 11.8, SciPy 1.11.4, Stable-Baselines3 2.2.1, and COMSOL Multiphysics 6.1 for the upstream vascular flow fields.

Install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps .
```

Install with conda:

```bash
conda env create -f environment.yml
conda activate ni-pinn
pip install --no-deps .
```

Build the container:

```bash
docker build -t ni-pinn:2.1.2 .
```

## Data

The verified access points are collected in `dataset_links.txt`.

TCGA-LIHC contains RNA-seq expression and clinical annotations for the 363-case cohort. Open GDC files are governed by NIH genomic data-sharing policies and prohibit participant re-identification. Controlled files require dbGaP authorization. The model uses a stage-stratified 254/36/73 split. Expression is transformed with log2(x + 1) and training-fitted quantile normalization. CIBERSORTx with LM22 produces 22 immune fractions, followed by an arcsine square-root transform. CIBERSORTx is available for non-commercial use under Stanford's service terms.

Synthetic NI-Bench is generated locally from the coupled system and therefore has no external download URL. It contains 10,000 twelve-week trajectories at 168 observation times, partitioned 8,000/1,000/1,000. Parameters are sampled from the ranges in the manuscript and integrated with RK45 at `rtol=1e-8` and `atol=1e-10`.

```bash
ni-pinn-prepare --output data --count 10000 --seed 0
```

The hepatic vascular cohort consists of 19 contrast-enhanced CT angiography geometries split 13/3/3. The article cited as its source does not expose a directly downloadable archive through a verified public endpoint, so no vascular URL is listed. Supply COMSOL 6.1 flow exports as NumPy archives containing `nodes`, `edges`, `radii`, `velocities`, and `tumor_center`.

Expected storage is approximately 650 MB for Synthetic NI-Bench, dependent on gzip compression, plus the selected GDC files and vascular meshes. Generate SHA-256 manifests after acquisition:

```bash
find data -type f -print0 | sort -z | xargs -0 sha256sum > data_manifest.sha256
```

## Training

The primary configuration matches the reported experiment: six hidden layers, 128 units per layer, Swish activations, 10,000 interior and 2,000 boundary collocation points per epoch, Adam at `1e-3`, cosine decay to `1e-6`, 1,200 epochs, batch size 256, early-stopping patience 100, and seeds 0–4. Loss weights are 1 for physics residuals, 10 for observations, 5 for neuro-immune constraints, and 1 for boundary conditions.

```bash
ni-pinn-train --config configs/main.yaml --data data/synthetic_ni_bench.h5 --output outputs/main
```

The nested schedule uses 100 ten-second fast steps, 42 four-hour medium steps, and 12 one-week slow steps. PPO uses 16 parallel environments, a 0.2 clipping ratio, discount 0.99, GAE 0.95, and 500,000 environment steps. The policy has three 256-unit hidden layers. Reward weights are 1.0 for navigation progress, 0.3 for therapy awareness, and 5.0 for collisions.

The experiment registry covers all five seeds for neuro-immune weights 0.1, 0.5, 1, 5, and 10; fast-step counts 50, 100, 200, and 500; Gaussian noise 0, 0.01, 0.05, 0.10, and 0.20; parameter shifts 0, 5, 10, 20, and 50 percent; observation dropout 0, 10, 30, 50, and 70 percent; and every reported baseline.

## Evaluation

```bash
ni-pinn-evaluate --config configs/main.yaml --data data/synthetic_ni_bench.h5 --output outputs/main
```

The target aggregate values across five seeds are 3.41 ± 0.18 percent immune-infiltration L2 relative error and 0.024 ± 0.003 tumor RMSE on TCGA-LIHC; 1.87 ± 0.12 percent coupled-system L2 relative error and 2.14 ± 0.19 percent parameter recovery error on Synthetic NI-Bench; and 92.8 ± 1.7 percent navigation efficiency with 0.847 ± 0.031 therapeutic efficacy on the vascular cohort.

Fast, medium, and slow subsystem errors are expected near 1.23 ± 0.11, 1.94 ± 0.16, and 2.78 ± 0.22 percent. Cross-patient evaluation targets 90.6 ± 2.2 percent navigation efficiency and 0.823 ± 0.037 therapeutic efficacy on validation geometries, and 88.2 ± 2.9 percent with 0.794 ± 0.044 on test geometries.

The evaluation module provides L2 relative error, RMSE, mean absolute error, parameter recovery error, Pearson correlation, navigation efficiency, therapeutic efficacy, bootstrap intervals, paired t-tests, Wilcoxon signed-rank tests, path-length normalization, collision rates, and perturbation degradation ratios.

## Compute budget

The reported run uses one NVIDIA A100 with 80 GB VRAM. NI-PINN contains approximately 1.47 million parameters, requires approximately 1.08 GFLOP per forward pass, reaches about 4.3 GB peak training memory at batch size 256, and takes about 9.4 seconds per epoch. The PINN stage takes approximately 8 hours and PPO navigation approximately 6 hours, for approximately 14 hours end to end. Dataset artifacts and run outputs should reserve at least 20 GB of disk space in addition to COMSOL project storage.

## Package map

`dynamics.py` defines the tumor, CD8, macrophage, MDSC, neurotransmitter, and drug residuals. `differential.py` provides temporal derivatives and spatial Laplacians. `losses.py` assembles physics, observation, pathway, boundary, and inter-scale terms. `multirate.py` runs nested temporal windows. `synthetic.py` creates NI-Bench trajectories. `preprocessing.py` isolates cohort normalization and stratified partitions. `vascular.py` defines graph-based flow lookup and microrobot mechanics. `navigation.py` exposes the therapy-aware environment. `ppo.py` performs clipped policy updates. `training.py` manages deterministic optimization and atomic state persistence. `reported_results.py` holds expected numerical values used for result comparison.

## Reproducibility controls

Each run seeds Python, NumPy, PyTorch, and CUDA from a single entry point. Training state stores the seed, model, optimizer, scheduler, configuration, epoch, and metrics. State writes use an atomic rename. Data transformations are fitted only on training partitions. Synthetic parameter truth is retained for evaluation and never passed into the solution network. Vascular partitions remain anatomically disjoint.

The default configuration is the full reported budget. Reduced settings should be treated as separate experiments and must not be compared with the aggregate reference values above.
