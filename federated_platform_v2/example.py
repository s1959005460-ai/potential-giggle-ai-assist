"""Example script demonstrating end-to-end usage (in-memory demo).

This script:
- registers default trainers/metrics
- starts server (in-process, not grpc)
- runs two local clients that perform one round each
- prints evaluation results (if evaluator provided)
"""
import logging, threading, time
from core.registry import register_trainer, register_metric
import trainer.register_defaults as _reg  # registers example trainers
import metrics.defaults as _mdefs       # registers default metrics
from federated.server_platform import FederatedService
from trainer.base import BaseTrainer
from trainer.gnn_trainer import GNNTrainer
from controller.epsilon_audit import EpsilonAudit
import torch
from torch.utils.data import TensorDataset, DataLoader
import yaml

# simple model builder
def build_model():
    import torch
    return torch.nn.Linear(4,1)

# small synthetic dataset
def make_loader(n=100, batch=16):
    import torch
    X = torch.randn(n,4); y = (torch.rand(n,1)>0.5).float()
    ds = TensorDataset(X,y)
    return DataLoader(ds, batch_size=batch, shuffle=True)

def main():
    cfg = yaml.safe_load(open('hydra_config/config.yaml'))
    logging.basicConfig(level=cfg['logging']['level'])
    # init model function returning numpy state_dict
    def init_model_fn():
        m = build_model()
        return {k: v.detach().cpu().numpy() for k,v in m.state_dict().items()}
    # create service (not starting gRPC server for demo)
    svc = FederatedService(cfg, model_store={'epsilon_audit': EpsilonAudit()}, init_model_fn=init_model_fn)
    # create two clients (in-process usage of FederatedService)
    # simulate client 1
    model1 = build_model()
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.1)
    trainer1 = GNNTrainer(model1, make_loader(50), None, opt1, torch.nn.BCEWithLogitsLoss())
    # client 2
    model2 = build_model()
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    trainer2 = GNNTrainer(model2, make_loader(60), None, opt2, torch.nn.BCEWithLogitsLoss())
    # clients produce updates and call server methods directly (no gRPC in demo)
    # client 1 state
    sd1 = {k: v.detach().cpu().numpy() for k,v in model1.state_dict().items()}
    sd2 = {k: v.detach().cpu().numpy() for k,v in model2.state_dict().items()}
    # push updates to server pending_updates and aggregate
    svc.pending_updates = [('c1', 0, 50, sd1, {}), ('c2', 0, 60, sd2, {})]
    svc.aggregate_and_update_global()
    print('Aggregated. Round:', svc.round_manager.get_round())
    print('Global model keys:', list(svc.global_model.keys()))
    # evaluate (no evaluator registered in this demo)
    print('Done.')
if __name__ == '__main__':
    main()
