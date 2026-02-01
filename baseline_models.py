from torch import nn
import torch
from util import BaseEncoder

class DeepSetBaseline(BaseEncoder):
    """1-body baseline: embed each point independently, no interactions."""
    def __init__(self, d_model):
        super().__init__(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
    def forward(self, x, _=None):
        B, N, _ = x.shape
        outs = []
        for b in range(B):
            h = self.embed(x[b])
            h = self.mlp(h)
            outs.append(self.predict(h))
        return torch.stack(outs)

class MPNNModel(BaseEncoder):
    """
    Baseline: Standard Message Passing (2-body interactions).
    Limited to a sparse budget of random edges.
    """
    def __init__(self, d_model):
        super().__init__(d_model)
        self.msg = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, edges):
        B, N, _ = x.shape
        outs = []
        for b in range(B):
            h = self.embed(x[b])
            m = self.msg(torch.cat([h[edges[:,0]], h[edges[:,1]]], dim=1))
            update = torch.zeros_like(h)
            update.index_add_(0, edges[:,0], m)
            h = self.norm(h + update)
            outs.append(self.predict(h))
        return torch.stack(outs)

class MPNN3Body(BaseEncoder):
    """
    Explicitly computes messages from Triplets (i, j, k).
    This matches the 'physics' of the curvature task better than the 2-body models.
    """
    def __init__(self, d_model):
        super().__init__(d_model)
        self.tri_msg = nn.Sequential(
            nn.Linear(3*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, triplets):
        B, N, _ = x.shape
        outs = []
        for b in range(B):
            h = self.embed(x[b])
            # Concatenate 3 features
            t_h = torch.cat([h[triplets[:,0]], h[triplets[:,1]], h[triplets[:,2]]], dim=1)
            msg = self.tri_msg(t_h)
            
            update = torch.zeros_like(h)
            update.index_add_(0, triplets[:,0], msg)
            h = self.norm(h + update)
            outs.append(self.predict(h))
        return torch.stack(outs)