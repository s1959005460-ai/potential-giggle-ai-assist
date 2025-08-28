# FedGNN Demo Package

This demo contains a configurable federated GNN framework with:
- Central DP (server-side)
- SCAFFOLD (gradient control variates)
- FedBN (batch-norm personalization)
- Heterogeneous client models + Knowledge Distillation hooks
- Communication compression (top-k, int8)
- Privacy accounting (Opacus RDP wrapper)
- Privacy monitoring utilities
- Robust aggregation (Trimmed Mean, Krum)
- Toy run script using TUDataset (e.g., MUTAG)

## Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run a smoke test (toy demo):

```bash
python run_experiment.py --dataset MUTAG --n_clients 5 --rounds 6 --device cpu --config config.yaml
```

3. Files of interest:
- `run_experiment.py` — main demo runner.
- `server.py`, `client.py`, `utils.py`, `pyg_utils.py` — core framework.
- `federated_privacy.py`, `privacy_monitor.py` — privacy accounting & monitoring.
- `analysis.py` — plotting & reporting utilities.
- `config.yaml` — default configuration.

Note: This demo expects PyTorch and PyTorch Geometric (PyG). If PyG installation is difficult, switch `model.type` in config to `mlp` for a CPU-only test.

