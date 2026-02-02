from torch import nn
from util import BaseEncoder
import torch

class Full2Trans(BaseEncoder):
    """
    Standard O(N^2) Self-Attention.
    Every point talks to every other point.
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__(d_model)
        # We use the standard PyTorch implementation for the "Classic" feel
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                     dim_feedforward=4*d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, _=None):
        # 1. Embed: [B, N, 2] -> [B, N, D]
        h = self.embed(x)
        
        # 2. Attention: [B, N, D] (Interacts all N^2 pairs)
        h = self.transformer(h)
        
        # 3. Predict
        outs = []
        for b in range(h.shape[0]):
            outs.append(self.predict(h[b]))
        return torch.stack(outs)

class Full3Trans(BaseEncoder):
    """
    THE FULL O(N^3) MODEL.
    Calculates interactions between ALL possible triplets (i, j, k).
    This is the theoretical upper bound (the 'Teacher').
    """
    def __init__(self, d_model, num_nodes):
        super().__init__(d_model)
        self.tri_msg = nn.Sequential(
            nn.Linear(3*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        
        # Generate ALL triplets
        # Warning: This grows N^3. 
        rng = torch.arange(num_nodes)
        i, j, k = torch.meshgrid(rng, rng, rng, indexing='ij')
        # Flatten to [N^3, 3]
        all_triplets = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)
        self.register_buffer('triplets', all_triplets)

    def forward(self, x, _=None):
        B, N, _ = x.shape
        outs = []
        triplets = self.triplets
        for b in range(B):
            h = self.embed(x[b])
            
            # This line is the memory killer: [N^3, 3*D]
            t_h = torch.cat([h[triplets[:,0]], h[triplets[:,1]], h[triplets[:,2]]], dim=1)
            
            msg = self.tri_msg(t_h)
            
            update = torch.zeros_like(h)
            
            # Since we sum over all N^3, the magnitude explodes if we don't scale.
            # Scaling by 1/N^2 is appropriate for averaging.
            scale = 1.0 / (N*N)
            update.index_add_(0, triplets[:,0], msg * scale)
            
            h = self.norm(h + update)
            outs.append(self.predict(h))
        return torch.stack(outs)