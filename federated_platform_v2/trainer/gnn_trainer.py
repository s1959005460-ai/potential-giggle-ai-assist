from trainer.base import BaseTrainer
import torch

class GNNTrainer(BaseTrainer):
    def train_one_epoch(self):
        self.model.train()
        for batch in self.train_loader:
            # expect batch to be appropriate for user's GNN (e.g., (graph, labels))
            xb, yb = batch
            xb = xb.to(self.device); yb = yb.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            self.optimizer.step()
