from torch import nn
from util import BaseEncoder
import torch

class Full2TransSeq(BaseEncoder):
    """
    O(N^2) Transformer with learned positional encodings.
    Unlike the Set-Transformer variant, this model is position-aware:
    the ordering of input points affects the output.
    """
    def __init__(self, d_model, num_nodes, num_heads=4):
        super().__init__(d_model)
        self.pos_embed = nn.Embedding(num_nodes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, _=None):
        B, N, _ = x.shape
        h = self.embed(x) + self.pos_embed.weight[:N]
        h = self.transformer(h)
        outs = []
        for b in range(B):
            outs.append(self.predict(h[b]))
        return torch.stack(outs)

class Full3TransSeq(BaseEncoder):
    """
    O(N^3) triplet interaction model with learned positional encodings.
    Position-aware variant of the Full3Trans set model.
    """
    def __init__(self, d_model, num_nodes):
        super().__init__(d_model)
        self.pos_embed = nn.Embedding(num_nodes, d_model)
        self.tri_msg = nn.Sequential(
            nn.Linear(3*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

        rng = torch.arange(num_nodes)
        i, j, k = torch.meshgrid(rng, rng, rng, indexing='ij')
        all_triplets = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)
        self.register_buffer('triplets', all_triplets)

    def forward(self, x, _=None):
        B, N, _ = x.shape
        outs = []
        triplets = self.triplets
        for b in range(B):
            h = self.embed(x[b]) + self.pos_embed.weight[:N]

            t_h = torch.cat([h[triplets[:,0]], h[triplets[:,1]], h[triplets[:,2]]], dim=1)
            msg = self.tri_msg(t_h)

            update = torch.zeros_like(h)
            scale = 1.0 / (N*N)
            update.index_add_(0, triplets[:,0], msg * scale)

            h = self.norm(h + update)
            outs.append(self.predict(h))
        return torch.stack(outs)
