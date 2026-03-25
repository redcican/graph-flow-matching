# Graph-Augmented Flow Matching (GAFM)

Official implementation of *Graph-Based Neural Network Augmented Flow Matching for Mixed-Type Tabular Data Generation*.

## Overview

GAFM integrates graph neural networks into geometric flow matching on product manifolds to model inter-feature and inter-sample dependencies for mixed-type tabular data generation.

The velocity field decomposes as:

$$v_\theta(x_t, t) = v_{\text{coord}} + v_{\text{feat}} + v_{\text{samp}}$$

where each component respects the product manifold structure $\mathcal{M} = \mathbb{R}^{d_c} \times \prod \Delta^{n_j-1} \times \prod S^1$ for continuous, categorical, and ordinal features respectively.

### Architecture

- **Coordinate-wise network** — independent MLPs per manifold type with Fisher-Rao geodesics for categorical and circular geodesics for ordinal features
- **Feature GNN** — static heterogeneous graph over columns with type-aware message passing (Einstein midpoint for categorical, circular mean for ordinal)
- **Sample GNN** — dynamic k-NN graph over batch samples for local trajectory coherence
- **Sinkhorn OT** — optimal transport pairing for straighter flow paths
- **Multi-stage training** — progressive unfreezing: coordinate-wise → +feature GNN → +sample GNN

## Installation

```bash
git clone https://github.com/redcican/graph-flow-matching.git
cd graph-flow-matching
pip install torch torch-geometric torchdiffeq
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Geometric
- torchdiffeq
- numpy, pandas, scikit-learn



## Package Structure

```
graph_flow_matching/
├── models/
│   ├── manifold_ops.py      # Geometric operations (Fisher-Rao, geodesics, distances)
│   ├── coord_net.py         # Coordinate-wise velocity networks
│   ├── feature_gnn.py       # Feature graph construction & GNN
│   ├── sample_gnn.py        # Dynamic sample graph & GNN
│   ├── velocity_field.py    # Main model (v_coord + v_feat + v_samp)
│   ├── ot_solver.py         # Sinkhorn optimal transport
│   ├── aggregation.py       # Einstein midpoint, circular mean
│   └── time_embedding.py    # Sinusoidal time encoding
├── training/
│   ├── trainer.py           # GAFMTrainer, DataPreprocessor, multi-stage training
│   └── sampler.py           # ODE-based sampling (dopri5 / Euler)
├── baselines/               # Unified wrappers for 9 baselines + GAFM
├── data/                    # Dataset loading utilities
├── evaluation/              # Metrics (W1, JSD, correlation error, DCR)
├── tests/                   # 152 pytest tests
└── utils/                   # Logging, seeding, device helpers
```

## Testing

```bash
python -m pytest graph_flow_matching/tests/ -v
```

152 tests covering manifold operations, model components, training pipeline, baselines, and end-to-end integration.

## Configuration

Key hyperparameters in `GAFMConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | MLP hidden dimension |
| `num_mlp_layers` | 5 | Layers per coordinate-wise MLP |
| `feat_gnn_layers` | 3 | Feature GNN message passing layers |
| `samp_gnn_layers` | 2 | Sample GNN layers |
| `k_neighbors` | 10 | k for dynamic sample k-NN graph |
| `sinkhorn_iterations` | 100 | OT solver iterations |
| `dependency_threshold` | 0.3 | Edge threshold for feature graph |
| `n_ode_steps` | 100 | ODE integration steps for sampling |
| `stage1_fraction` | 0.2 | Training budget for coordinate-wise warmup |
| `stage2_fraction` | 0.3 | Training budget for +feature GNN |
| `stage3_fraction` | 0.5 | Training budget for full model |

## License

MIT
