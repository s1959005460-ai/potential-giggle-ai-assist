import argparse
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import os
from copy import deepcopy
import yaml
import json
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pyg_utils import get_model\nfrom federated_privacy import FederatedPrivacyAuditor\nfrom privacy_monitor import compute_gradient_norms, summarize_leakage_risk
from client import Client
from server import Server
from utils import vector_to_state_dict

def split_dataset(dataset, n_clients=5, non_iid=False, test_ratio=0.2):
    n = len(dataset)
    indices = list(range(n))
    if non_iid:
        labels = [int(d.y.item()) for d in dataset]
        idx_by_label = {}
        for i, lab in enumerate(labels):
            idx_by_label.setdefault(lab, []).append(i)
        shards = [[] for _ in range(n_clients)]
        lab_keys = list(idx_by_label.keys())
        for i, lab in enumerate(lab_keys):
            targets = idx_by_label[lab]
            for j, idx in enumerate(targets):
                shards[j % n_clients].append(idx)
    else:
        random.shuffle(indices)
        chunk = n // n_clients
        shards = [indices[i*chunk:(i+1)*chunk] for i in range(n_clients)]
        remainder = n % n_clients
        idx = chunk * n_clients
        for i in range(remainder):
            shards[i].append(indices[idx + i])
    # split each shard into train/test
    client_shards = []
    for s in shards:
        random.shuffle(s)
        cut = max(1, int(len(s) * (1 - test_ratio)))
        train_idx = s[:cut]
        test_idx = s[cut:]
        client_shards.append({'train': train_idx, 'test': test_idx})
    return client_shards

def build_clients_from_dataset(config, dataset, device, n_clients=5, non_iid=False, malicious_ids=None):
    shards = split_dataset(dataset, n_clients=n_clients, non_iid=non_iid)
    clients = []
    for i in range(n_clients):
        train_list = [dataset[idx] for idx in shards[i]['train']]
        test_list = [dataset[idx] for idx in shards[i]['test']]
        is_mal = (malicious_ids is not None and i in malicious_ids)
        c = Client(client_id=i, config=config, device=device, data_list=train_list, malicious=is_mal)
        c.test_list = test_list
        c.get_model(config)
        clients.append(c)
    return clients

def evaluate_personalized(clients):
    results = {}
    for c in clients:
        # evaluate client's local model on its test_list
        if getattr(c, 'test_list', None) is None or len(c.test_list) == 0:
            continue
        loader = DataLoader(c.test_list, batch_size=16)
        c.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for data in loader:
                data = data.to(c.device)
                out = c.model(data)
                y = data.y
                if out.dim() > 1 and out.shape[0] != y.shape[0]:
                    out = out.view(y.shape[0], -1)
                pred = out.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                labels.extend(y.cpu().numpy().tolist())
        if len(labels) == 0:
            continue
        results[c.client_id] = {
            'acc': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='macro')
        }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MUTAG')
    parser.add_argument('--n_clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', default=None, help='path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = args.device
    root = os.path.join('data', args.dataset)
    dataset = TUDataset(root, name=args.dataset)
    print(f'Loaded dataset {args.dataset}, #graphs={len(dataset)}, num_classes={dataset.num_classes}')

    global_model = get_model(config)
    server = Server(global_model, config)
    auditor = None
    if config.get('dp', {}).get('enabled', False):
        try:
            auditor = FederatedPrivacyAuditor(target_delta=config.get('privacy', {}).get('delta', 1e-5))
        except Exception as e:
            print('Privacy auditor not available:', e)

    clients = build_clients_from_dataset(config, dataset, device, n_clients=args.n_clients, non_iid=config.get('clients',{}).get('non_iid', False), malicious_ids=[1])

    # resume if provided
    start_round = 0
    if args.resume is not None:
        server.load_checkpoint(args.resume)
        start_round = server.round

    for c in clients:
        c.set_weights(server.global_model.state_dict())

    test_loader = DataLoader(dataset, batch_size=16)

    metrics_log_path = os.path.join('logs', 'metrics.jsonl')
    os.makedirs('logs', exist_ok=True)

    for r in range(start_round, args.rounds):
        print(f'=== Round {r+1} ===')
        selected_idx = server.select_clients(clients, num_clients=max(1, len(clients)//2), seed=42)
        selected_clients = [clients[i] for i in selected_idx]
        print('Selected clients:', selected_idx)

        server_c = server.get_c_global_for_client(device) if config.get('use_scaffold', False) else None

        client_updates = []
        client_counts = []
        for c in selected_clients:
            c.set_weights(server.global_model.state_dict())
            res = c.train_local(server_c_global=server_c, global_state=server.global_model.state_dict(), round_index=r)
            client_updates.append(res)
            client_counts.append(res.get('num_samples', len(c.data_list)))

        server.aggregate(client_updates, client_counts=client_counts, round_index=r)

        # save checkpoint
        server.save_checkpoint(r)

        # evaluate global model
        metrics = server.evaluate(test_loader)
        print(f"Global eval: loss={metrics['loss']:.4f}, acc={metrics['acc']:.4f}")

        # personalized evaluation
        personal = evaluate_personalized(clients)

        # compute client contribution stats (norms, similarities)
        norms = []
        for up in client_updates:
            if 'delta' in up:
                norms.append(float(up['delta'].norm().item()))
            elif 'delta_compressed' in up:
                norms.append(None)
            else:
                norms.append(None)

        # log metrics to jsonl
        log = {
            'round': r,
            'global': metrics,
            'personal': personal,
            'client_counts': client_counts,
            'client_norms': norms,
            'selected_clients': selected_idx,
            'epsilon': eps,
            'leak_summary': leak_summary
        }
        with open(metrics_log_path, 'a') as f:
            f.write(json.dumps(log) + '\n')

    print('Training finished.')

if __name__ == '__main__':
    main()
