"""Independent, checkpoint-preserving adaptations of our existing models."""
import torch
from torch import nn
from sequence_compare import make_model, FEATURES


class StrategyModel(nn.Module):
    def __init__(self, architecture, strategy):
        super().__init__()
        self.base = make_model(architecture, len(FEATURES))
        self.strategy = strategy
        if strategy == 'categorical':
            self.condition = nn.Linear(6, len(FEATURES), bias=False)
            nn.init.zeros_(self.condition.weight)
        if strategy == 'cnn':
            self.front = nn.Sequential(nn.Conv1d(len(FEATURES),32,3,padding=1),nn.ReLU(),
                                       nn.Conv1d(32,len(FEATURES),3,padding=1))
            nn.init.zeros_(self.front[-1].weight); nn.init.zeros_(self.front[-1].bias)
        if strategy == 'delta':
            self.delta_head = nn.Linear(112,1)

    def forward(self, x, rc, auxiliary=False):
        if self.strategy == 'categorical':
            x = x + self.condition(rc).unsqueeze(1)
        if self.strategy == 'cnn':
            x = x + self.front(x.transpose(1,2)).transpose(1,2)
        if self.strategy == 'delta':
            hidden = self.base.lstm(x)[0]
            pred = self.base.head(hidden).squeeze(-1)
            return (pred,self.delta_head(hidden).squeeze(-1)) if auxiliary else pred
        return self.base(x)


def objective(pred, target, mask, strategy, delta_pred=None):
    if strategy == 'mse_control':
        return (pred-target).square().mean()
    primary = (pred[mask]-target[mask]).abs().mean()
    if strategy != 'delta':
        return primary
    # Only adjacent pairs within the same breath's inspiratory phase contribute.
    pairs = mask[:,1:] & mask[:,:-1]
    delta = target[:,1:]-target[:,:-1]
    aux = (delta_pred[:,1:][pairs]-delta[pairs]).abs().mean()
    return primary + .2*aux
