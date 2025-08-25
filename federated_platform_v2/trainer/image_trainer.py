from trainer.base import BaseTrainer
import torch

class ImageTrainer(BaseTrainer):
    def train_one_epoch(self):
        self.model.train()
        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            self.optimizer.step()
