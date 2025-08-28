import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def compute_gradient_norms(updates):
    norms = [u.norm().item() for u in updates]
    return norms

def compute_update_similarity_matrix(updates):
    mats = torch.stack([u / (u.norm()+1e-12) for u in updates]).cpu().numpy()
    sim = cosine_similarity(mats)
    return sim

def summarize_leakage_risk(updates, threshold_sim=0.95):
    sim = compute_update_similarity_matrix(updates)
    n = sim.shape[0]
    high_pairs = np.where(np.triu(sim, k=1) > threshold_sim)
    suspicious_pairs = list(zip(high_pairs[0].tolist(), high_pairs[1].tolist()))
    return {'num_suspicious_pairs': len(suspicious_pairs), 'pairs': suspicious_pairs, 'similarity_matrix': sim}
