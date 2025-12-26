# Genomic Prediction with Graph Neural Networks

This repository contains the code and analysis for a specialization project investigating **Graph Neural Networks (GNNs) for genomic prediction** in house sparrows (*Passer domesticus*). The study compares GNN-based models (GCN, GraphSAGE) with traditional genomic prediction methods (MLP, GBLUP) for predicting morphological traits across island populations in the Helgeland archipelago.

## Methods Overview

### Cross-Validation Strategies

1. **Leave-One-Island-Out (LOIO)**: Tests generalization across populations by holding out entire islands as test sets.
2. **Within-Population**: Standard k-fold CV within each island to assess within-population prediction.

### Models

- **MLP**: Fully-connected neural network baseline (no graph structure)
- **GCN**: Graph Convolutional Network using genomic relatedness
- **GraphSAGE**: Sampling-based GNN with neighborhood aggregation
- **GBLUP**: Genomic Best Linear Unbiased Prediction (traditional method)

### Graph Construction

Graphs are constructed from the **Genomic Relationship Matrix (GRM)**:
- **kNN graphs**: k-nearest neighbors based on genetic similarity
- **Cutoff graphs**: Threshold-based edges on GRM values

### Phenotypes

- Tarsus length (`thr_tarsus`)
- Body mass (`body_mass`)
- Wing length (`thr_wing`)

## Usage

### Running Nested Cross-Validation

```bash
# Example: Run nested CV for GCN on body mass
python -m src.nested_cv --config configs/config_nested_within_gcn_mass.json
```

### Configuration

Experiments are configured via JSON files in `configs/`. Key sections:
- `paths`: Input/output file locations
- `model`: Architecture hyperparameters (hidden dims, dropout, etc.)
- `training`: Learning rate, optimizer, epochs
- `cv`: Cross-validation settings (outer/inner folds)
- `search_space`: Optuna hyperparameter search ranges

### Running on HPC (SLURM)

```bash
sbatch src/SLURM/nested_cv.slurm
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- Optuna (hyperparameter tuning)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (visualization)

## Analysis Notebooks

The `analysis/` folder contains Jupyter notebooks that generate all thesis figures:

1. **`nested_cv_analysis.ipynb`**: Main results comparing model performance across traits and validation scenarios
2. **`hyperparameter_graph_analysis.ipynb`**: Analysis of graph construction choices (kNN vs cutoff, SNP selection)
3. **`results_step1_summary.ipynb`**: Phenotype distributions and data exploration