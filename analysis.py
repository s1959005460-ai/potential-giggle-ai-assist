import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_accuracy_curves(results, out='accuracy.png'):
    plt.figure()
    for k, v in results.items():
        plt.plot(v, label=k)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out)
    print('Saved', out)

def save_results_json(results, path='results.json'):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved', path)

def plot_similarity_heatmap(sim_matrix, out='sim_heatmap.png'):
    plt.figure(figsize=(6,6))
    plt.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Client Update Cosine Similarity')
    plt.savefig(out)
    print('Saved', out)

def plot_tsne_placeholder(features, labels=None, out='tsne.png'):
    try:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2).fit_transform(features)
        plt.figure()
        if labels is None:
            plt.scatter(emb[:,0], emb[:,1], s=5)
        else:
            for lab in np.unique(labels):
                idx = labels == lab
                plt.scatter(emb[idx,0], emb[idx,1], s=5, label=str(lab))
            plt.legend()
        plt.title('t-SNE of client updates')
        plt.savefig(out)
        print('Saved', out)
    except Exception as e:
        print('t-SNE plotting failed (sklearn may be missing):', e)
