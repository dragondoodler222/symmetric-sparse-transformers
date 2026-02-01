from torch import nn
from util import BaseEncoder
import torch

class Sparse2Trans(BaseEncoder):
    """ Used for both DeBruijn and Sparse 2-Body """
    def __init__(self, d_model, edge_source=None, edge_target=None):
        super().__init__(d_model)
        self.msg = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

        # Register static edges if provided at init
        if edge_source is not None and edge_target is not None:
            self.register_buffer('src', edge_source)
            self.register_buffer('dst', edge_target)
            # Compute degree normalization: 1/degree for each source node
            num_nodes = max(edge_source.max(), edge_target.max()) + 1
            degree = torch.zeros(num_nodes)
            degree.index_add_(0, edge_source, torch.ones_like(edge_source, dtype=torch.float))
            degree = degree.clamp(min=1)
            inv_degree = 1.0 / degree
            self.register_buffer('_inv_degree', inv_degree[edge_source])
        else:
            self.src = None
            self.dst = None
            self._inv_degree = None

    def forward(self, x, edges=None):
        B, N, _ = x.shape
        outs = []

        if edges is not None:
            src = edges[:, 0]
            dst = edges[:, 1]
            # Compute degree normalization on the fly
            degree = torch.zeros(N, device=x.device)
            degree.index_add_(0, src, torch.ones(src.shape[0], device=x.device))
            degree = degree.clamp(min=1)
            inv_deg = (1.0 / degree)[src]
        else:
            src = self.src
            dst = self.dst
            inv_deg = self._inv_degree

        for b in range(B):
            h = self.embed(x[b])
            # Message Passing
            m = self.msg(torch.cat([h[src], h[dst]], dim=1))
            update = torch.zeros_like(h)
            update.index_add_(0, src, m * inv_deg.unsqueeze(1))
            h = self.norm(h + update)
            outs.append(self.predict(h))
        return torch.stack(outs)

class Sparse3Trans(BaseEncoder):
    """
    Sparse 3-Body: Takes triplets either at construction or at forward time.
    """
    def __init__(self, d_model, triplets=None):
        super().__init__(d_model)
        self.tri_msg = nn.Sequential(
            nn.Linear(3*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

        if triplets is not None:
            self.register_buffer('triplets', triplets)
        else:
            self.triplets = None

    def forward(self, x, triplets=None):
        B, N, _ = x.shape
        outs = []
        trips = triplets if triplets is not None else self.triplets
        for b in range(B):
            h = self.embed(x[b])
            t_h = torch.cat([h[trips[:,0]], h[trips[:,1]], h[trips[:,2]]], dim=1)
            msg = self.tri_msg(t_h)
            update = torch.zeros_like(h)
            update.index_add_(0, trips[:,0], msg)
            h = self.norm(h + update)
            outs.append(self.predict(h))
        return torch.stack(outs)
