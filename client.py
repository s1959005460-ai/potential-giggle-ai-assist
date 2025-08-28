import torch
from torch_geometric.loader import DataLoader
from utils import params_to_vector, vector_to_params, state_dict_to_vector, vector_to_state_dict, is_bn_key, topk_compress, quantize_int8, compute_mask_sum_for_client

class Client:
    def __init__(self, client_id, config, device='cpu', data_list=None, malicious=False, compute_capacity=1.0, model_type=None):
        self.client_id = client_id
        self.config = config
        self.device = torch.device(device)
        self.malicious = malicious
        self.compute_capacity = compute_capacity
        self.model = None
        self.c_local = None
        self.personalized = None
        self.data_list = data_list or []
        self.dataloader = None
        self.model_type = model_type
        self._build_dataloader()

    def _build_dataloader(self):
        batch_size = int(self.config.get('local', {}).get('batch_size', 8))
        if len(self.data_list) == 0:
            self.dataloader = []
        else:
            self.dataloader = DataLoader(self.data_list, batch_size=batch_size, shuffle=True)

    def get_model(self, config):
        from pyg_utils import get_model
        if self.model_type:
            cfg = dict(config)
            cfg['model'] = dict(cfg.get('model', {}))
            cfg['model']['type'] = self.model_type
            self.model = get_model(cfg).to(self.device)
        else:
            self.model = get_model(config).to(self.device)
        self.c_local = {k: torch.zeros_like(v).to(self.device) for k, v in self.model.state_dict().items()}
        return self.model

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def simulate_malicious(self, flat_update):
        if not self.malicious:
            return flat_update
        atk = self.config.get('attack', {}).get('type', 'signflip')
        if atk == 'signflip':
            return -5.0 * flat_update
        elif atk == 'random':
            return torch.randn_like(flat_update) * flat_update.norm()
        return flat_update

    def train_local(self, server_c_global=None, global_state=None, public_loader=None, round_index=0):
        cfg = self.config
        local_cfg = cfg.get('local', {})
        epochs = int(local_cfg.get('epochs', 1) * self.compute_capacity)
        lr = local_cfg.get('lr', 0.01)
        local_steps = int(local_cfg.get('local_steps', 1))
        use_sca = cfg.get('use_scaffold', False)
        use_pf = cfg.get('pFedMe', {}).get('enabled', False)\n        # pFedMe params\n        pf_lam = float(cfg.get('pFedMe', {}).get('lam', 15.0))\n        pf_beta = float(cfg.get('pFedMe', {}).get('beta', 0.01))\n        pf_local_steps = int(cfg.get('pFedMe', {}).get('local_steps', 5))
        # FedProx mu
        mu = float(cfg.get('fedprox', {}).get('mu', 0.0))

        opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        w_global_vec = params_to_vector(self.model).detach().clone()
        total_steps = 0
        for ep in range(epochs):
            if not self.dataloader:
                break
            for batch in self.dataloader:
                if torch.rand(1).item() > self.compute_capacity:
                    continue
                batch = batch.to(self.device)
                opt.zero_grad()
                out = self.model(batch)
                loss = loss_fn(out, batch.y)
                # FedProx proximal term: mu/2 * ||w - w_global||^2
                if mu > 0:
                    w_vec = params_to_vector(self.model)
                    prox = (mu / 2.0) * torch.norm(w_vec - w_global_vec) ** 2
                    loss = loss + prox
                if use_pf and self.personalized is not None:
                    w_vec = params_to_vector(self.model)
                    theta = self.personalized.to(self.device)
                    lam = cfg.get('pFedMe', {}).get('lam', 15.0)
                    loss = loss + (lam / 2.0) * torch.norm(w_vec - theta) ** 2
                loss.backward()
                if use_sca and server_c_global is not None:
                    vec_sg = state_dict_to_vector(server_c_global).to(self.device)
                    vec_cl = state_dict_to_vector(self.c_local).to(self.device)
                    diff = (vec_sg - vec_cl)
                    idx = 0
                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        numel = p.numel()
                        add_chunk = diff[idx: idx + numel].view_as(p.data)
                        p.grad.data.add_(add_chunk)
                        idx += numel
                opt.step()
                total_steps += 1
                if total_steps >= local_steps:
                    break

        w_local_vec = params_to_vector(self.model).detach().clone()
        delta = (w_local_vec - w_global_vec).detach()
        # pFedMe theta update: move personalized vector toward w_local\n        if use_pf:\n            theta = self.personalized\n            theta = theta - pf_beta * (theta - w_local_vec)\n            self.personalized = theta.clone().detach()\n
        # compute delta_state (per-key) to support FedBN
        local_state = self.model.state_dict()
        delta_state = {}
        for k in local_state:
            delta_state[k] = (local_state[k].detach().cpu() - global_state[k].detach().cpu())

        # SCAFFOLD delta_c
        delta_c = None
        if use_sca:
            denom = max(1e-12, lr * max(1, total_steps))
            delta_c_vec = (w_global_vec - w_local_vec) / denom
            dc_state = vector_to_state_dict(delta_c_vec, self.model.state_dict())
            if server_c_global is not None:
                for k in self.c_local:
                    self.c_local[k] = self.c_local[k] + dc_state[k].to(self.device) - server_c_global[k].to(self.device)
            delta_c = delta_c_vec.clone()

        # Secure aggregation: add pairwise masks if enabled
        if cfg.get('secure_agg', {}).get('enabled', False):
            n_clients = cfg.get('clients', {}).get('n_clients', 5)
            shared_seed = cfg.get('secure_agg', {}).get('shared_seed', 12345)
            mask = compute_mask_sum_for_client(n_clients, delta.shape, self.client_id, round_index, shared_seed=shared_seed, device=delta.device)
            delta = delta + mask
            # note: server will see masked deltas and masks will cancel when summed

        # apply compression if configured
        comm = cfg.get('comm', {})
        if comm.get('compress') == 'topk':
            k = max(1, int(comm.get('topk_ratio', 0.01) * delta.numel()))
            comp = topk_compress(delta, k)
            ret = {'client_id': self.client_id, 'delta_compressed': comp, 'delta_state': delta_state}
        elif comm.get('compress') == 'int8':
            ret = {'client_id': self.client_id, 'delta_compressed': quantize_int8(delta), 'delta_state': delta_state}
        else:
            # Local DP option: add noise before sending (LDP)
            if cfg.get('local_dp', {}).get('enabled', False):
                noise_scale = float(cfg.get('local_dp', {}).get('noise_scale', 1.0))
                noise = torch.normal(0.0, noise_scale, size=delta.shape, device=delta.device)
                delta = delta + noise
            ret = {'client_id': self.client_id, 'delta': delta, 'delta_state': delta_state}

        if delta_c is not None:
            ret['delta_c'] = delta_c
        ret['num_samples'] = len(self.data_list)
        return ret

    def train(self, server_c_global=None, global_state=None, round_index=0):
        result = self.train_local(server_c_global=server_c_global, global_state=global_state, round_index=round_index)
        stats = {'num_samples': len(self.data_list)}
        return result, stats
