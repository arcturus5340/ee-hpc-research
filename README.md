# EE-HPC Research

Exploratory and analytical workflows on ee-cpt generated CSV data.

## 🔍 Overview

This repository contains Jupyter notebooks and CSV datasets generated using the ee-cpt library. The focus is on analyzing performance and energy efficiency metrics from HPC applications, specifically:

- **ICON**: A global atmospheric model.
- **LULESH**: A proxy application for unstructured Lagrangian hydrodynamics.
- **NPB-MZ-LU**: A variant of the NAS Parallel Benchmarks focusing on LU decomposition.

## 📂 Repository Structure

- `data/`: Raw CSV datasets generated via ee-cpt.
- `jube_configs/`: JUBE configuration files for running experiments.
- `notebooks/`: Jupyter notebooks for data exploration and analysis.
- `scripts/`: Scripts for emulating POP metrics calculation from raw data.
- `results/`: Processed visualizations.
- `README.md`: This file :)

## 🚀 Usage

1. Clone the repository:

```
git clone https://github.com/arcturus5340/ee-hpc-research.git
cd ee-hpc-research
```

2. Navigate to the `notebooks/` directory and launch Jupyter Notebook:

```
cd notebooks
jupyter notebook
```

3. Follow the instructions within each notebook to reproduce the analyses.
