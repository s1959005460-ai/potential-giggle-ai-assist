from core.registry import register_trainer
from trainer.gnn_trainer import GNNTrainer
from trainer.image_trainer import ImageTrainer

register_trainer('gnn')(GNNTrainer)
register_trainer('image')(ImageTrainer)
