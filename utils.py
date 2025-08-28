import torch
import math
import numpy as np

from torch.nn.utils import parameters_to_vector, vector_to_parameters

def params_to_vector(model):
    return parameters_to_vector([p for p in model.parameters()]).detach()

def vector_to_params(vec, model):
    vector_to_parameters(vec, [p for p in model.parameters()])

def clip_tensor_by_norm(tensor, max_norm):
    norm = torch.norm(tensor)
    if norm > max_norm:
        return tensor * (max_norm / (norm + 1e-12))
    return tensor

def clip_update_vector(update_vec, max_norm):
    return clip_tensor_by_norm(update_vec, max_norm)

def add_gaussian_noise(tensor, std, device=None):
    if std <= 0:
        return tensor
    if device is None:
        device = tensor.device
    noise = torch.normal(0.0, std, size=tensor.shape, device=device)
    return tensor + noise

# Robust aggregators
def trimmed_mean(updates, trim_ratio=0.1):
    stacked = torch.stack(updates, dim=0)
    n, d = stacked.shape
    k = int(math.floor(trim_ratio * n))
    if k == 0:
        return torch.mean(stacked, dim=0)
    sorted_vals, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_vals[k:n-k, :]
    return torch.mean(trimmed, dim=0)

def krum(updates, f=1):
    n = len(updates)
    if n <= 2*f + 2:
        return torch.mean(torch.stack(updates), dim=0)
    distances = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j: continue
            dists.append(torch.sum((updates[i] - updates[j]) ** 2).item())
        dists.sort()
        m = n - f - 2
        distances.append(sum(dists[:m]))
    best = int(np.argmin(distances))
    return updates[best]

# state dict helpers
def state_dict_to_vector(state_dict, device=None):
    parts = []
    for k in sorted(state_dict.keys()):
        v = state_dict[k].detach().flatten()
        if device:
            v = v.to(device)
        parts.append(v)
    return torch.cat(parts)

def vector_to_state_dict(vec, state_dict_template):
    out = {}
    idx = 0
    for k in sorted(state_dict_template.keys()):
        shape = state_dict_template[k].shape
        numel = state_dict_template[k].numel()
        out[k] = vec[idx: idx + numel].view(shape).clone()
        idx += numel
    return out

def is_bn_key(key: str):
    k = key.lower()
    return ('bn' in k) or ('batchnorm' in k) or ('running_mean' in k) or ('running_var' in k) or ('num_batches_tracked' in k)

# compression utilities
def topk_compress(vec, k):
    if k <= 0 or k >= vec.numel():
        return None
    vals, idx = torch.topk(vec.abs(), k)
    mask_idx = idx
    kept_vals = vec[mask_idx].cpu().numpy()
    kept_idx = mask_idx.cpu().numpy()
    return {'indices': kept_idx, 'values': kept_vals, 'shape': vec.shape}

def topk_decompress(compressed):
    if compressed is None:
        return None
    shape = compressed['shape']
    vec = torch.zeros(shape, dtype=torch.float32)
    vec[compressed['indices']] = torch.from_numpy(compressed['values'])
    return vec

def quantize_int8(vec):
    vmax = vec.abs().max().item()
    if vmax == 0:
        return {'scale': 1.0, 'q': vec.cpu().numpy().astype('int8')}
    scale = vmax / 127.0
    q = (vec.cpu().numpy() / scale).round().astype('int8')
    return {'scale': scale, 'q': q}

def dequantize_int8(qdict):
    import torch
    return torch.from_numpy(qdict['q'].astype('float32')) * qdict['scale']


# ---------------- Secure aggregation mask helpers ----------------
def pairwise_mask(shape, i, j, round_index, shared_seed=12345, dtype=torch.float32, device='cpu'):
    '''
    Deterministic pseudo-random mask shared between client i and j for given round.
    Convention: client with smaller id adds mask, larger subtracts mask.
    '''
    import numpy as np
    seed = (int(shared_seed) * 1000003) ^ (int(round_index) * 9176) ^ (int(i) * 1009) ^ (int(j) * 917)
    rng = np.random.RandomState(seed)
    vals = rng.normal(loc=0.0, scale=1.0, size=int(np.prod(shape))).astype('float32')
    arr = torch.from_numpy(vals).view(shape).to(device=device, dtype=dtype)
    return arr

def compute_mask_sum_for_client(n_clients, shape, client_id, round_index, shared_seed=12345, device='cpu'):
    '''
    Sum of masks to be added by this client so that aggregated masks cancel out on server.
    For all pairs (i,j) with i<j: client i adds +R_ij, client j adds -R_ij.
    For client 'client_id', compute sum_{j != client_id} sign(client_id,j) * R_{min,max}
    '''
    total = torch.zeros(shape, dtype=torch.float32, device=device)
    for other in range(n_clients):
        if other == client_id: continue
        i, j = (client_id, other) if client_id < other else (other, client_id)
        mask = pairwise_mask(shape, i, j, round_index, shared_seed=shared_seed, device=device)
        # if client_id < other -> this client adds +mask, else subtract
        sign = 1.0 if client_id < other else -1.0
        total = total + sign * mask
    return total
