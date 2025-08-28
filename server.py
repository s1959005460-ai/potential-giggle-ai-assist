import random
import torch
from utils import params_to_vector, vector_to_params, trimmed_mean, krum, is_bn_key, vector_to_state_dict
from torch.nn.functional import cross_entropy
import os

class Server:
    def __init__(self, global_model, config, checkpoint_dir='checkpoints'):
        self.global_model = global_model
        self.config = config
        self.c_global = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        self.device = next(global_model.parameters()).device
        self.round = 0
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # server optimizer for FedOPT
        opt_cfg = config.get('server_optimizer', {'type': 'sgd', 'lr': 1.0})
        self.server_opt = None
        if opt_cfg.get('type', 'sgd').lower() == 'adam':
            self.server_opt = torch.optim.Adam(self.global_model.parameters(), lr=float(opt_cfg.get('lr', 1.0)))
        else:
            # default small lr SGD to apply aggregated updates
            self.server_opt = torch.optim.SGD(self.global_model.parameters(), lr=float(opt_cfg.get('lr', 1.0)))

    def select_clients(self, clients, num_clients, seed=None):
        if seed is not None:
            random.seed(seed + self.round)
        n = len(clients)
        num = min(num_clients, n)
        return random.sample(range(n), num)

    def aggregate(self, client_updates, client_counts=None, round_index=None):
        cfg = self.config.get('dp', {})
        robust_cfg = self.config.get('robust', {})
        device = self.device
        template = self.global_model.state_dict()
        accum = {k: torch.zeros_like(v).to(device) for k,v in template.items() if not is_bn_key(k)}
        n_clients = len(client_updates)
        # collect accumulators in state-dict space
        for up in client_updates:
            if 'delta_state' in up:
                delta_state = up['delta_state']
            else:
                delta_state = vector_to_state_dict(up['delta'].to(device), template)
            for k in accum:
                accum[k] += delta_state[k].to(device)
        for k in accum:
            accum[k] = accum[k] / max(1, n_clients)
        # DP noise
        if cfg.get('enabled', False):
            noise_multiplier = float(cfg.get('noise_multiplier', 1.0))
            clip = float(cfg.get('clip_norm', 1.0))
            for k in accum:
                noise = torch.normal(0.0, noise_multiplier * clip, size=accum[k].shape, device=device)
                accum[k] = accum[k] + noise
        # If using server optimizer (FedOPT), treat -accum as pseudo-gradient and step optimizer
        if self.server_opt is not None:
            # set grads manually
            self.server_opt.zero_grad()
            with torch.no_grad():
                for name, p in self.global_model.named_parameters():
                    if is_bn_key(name): 
                        # skip BN parameters from optimization (FedBN)
                        p.grad = None
                        continue
                    # find corresponding tensor in accum by key name
                    if name in accum:
                        # pseudo-gradient is negative of accumulated delta
                        g = -accum[name]
                        if p.grad is None:
                            p.grad = g.clone()
                        else:
                            p.grad.copy_(g)
            # step optimizer
            self.server_opt.step()
        else:
            # fallback: directly apply accum
            global_state = self.global_model.state_dict()
            for k in global_state:
                if not is_bn_key(k):
                    global_state[k] = global_state[k].to(device) + accum[k]
            self.global_model.load_state_dict(global_state)
        # update SCAFFOLD control variate if present
        if any('delta_c' in u for u in client_updates):
            all_delta_c = [u['delta_c'].to(device) for u in client_updates if 'delta_c' in u]
            avg_dc = torch.mean(torch.stack(all_delta_c), dim=0)
            dc_state = vector_to_state_dict(avg_dc, template)
            for k in self.c_global:
                self.c_global[k] = self.c_global[k].to(device) + dc_state[k].to(device)
        self.round += 1

    def get_c_global_for_client(self, device):
        return {k: v.clone().to(device) for k, v in self.c_global.items()}

    def save_checkpoint(self, round_index):
        path = os.path.join(self.checkpoint_dir, f'checkpoint_round_{round_index}.pt')
        ckpt = {
            'round': round_index,
            'model_state': self.global_model.state_dict(),
            'c_global': self.c_global,
        }
        if self.server_opt is not None:
            ckpt['server_opt_state'] = self.server_opt.state_dict()
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path):
        data = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(data['model_state'])
        self.c_global = {k: v.to(self.device) for k, v in data.get('c_global', {}).items()}
        if 'server_opt_state' in data and self.server_opt is not None:
            self.server_opt.load_state_dict(data['server_opt_state'])
        self.round = int(data.get('round', 0))

    def evaluate(self, dataloader):
        self.global_model.eval()
        total, correct = 0, 0
        loss_sum = 0.0
        device = self.device
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                out = self.global_model(data)
                y = data.y
                if out.dim() > 1 and out.shape[0] != y.shape[0]:
                    out = out.view(y.shape[0], -1)
                loss = cross_entropy(out, y)
                loss_sum += loss.item() * y.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return {'loss': loss_sum / (total + 1e-12), 'acc': correct / (total + 1e-12)}
