import torch
from torch import nn
from pytorch_metric_learning import  miners, distances, reducers, losses

class ReIDCrossEntropyLoss(nn.Module):
    def __init__(self, reid_dim, num_classes):
        super().__init__()
        # 2-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(reid_dim, reid_dim, bias=False),
            nn.BatchNorm1d(reid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reid_dim, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        if self.training:
            logits = self.classifier(embeddings)
            loss = self.loss_fn(logits, labels.long())
        else:
            # loss in evaluation mode is zero
            loss = torch.zeros_like(labels, device=embeddings.device)

        return loss

class ReIDTripletLoss(nn.Module):
    def __init__(self, reid_dim, miner=None):
        super().__init__()
        distance = distances.CosineSimilarity()
        self.loss_fn = losses.TripletMarginLoss(distance=distance)
        self.mining_fn = None if miner is None else miners.TripletMarginMiner(distance=distance, type_of_triplets=miner)

    def forward(self, embeddings, labels):
        if self.training and self.mining_fn is not None:
            triplets = self.mining_fn(embeddings, labels)
            loss = self.loss_fn(embeddings, labels, triplets)
        
        # in evaluation mode, don't use mining
        else:
            loss = self.loss_fn(embeddings, labels)
        
        return loss
