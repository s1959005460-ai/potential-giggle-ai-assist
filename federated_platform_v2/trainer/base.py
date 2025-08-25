from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, loss_fn=None, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError
