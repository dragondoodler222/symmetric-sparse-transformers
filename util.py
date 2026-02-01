from torch import nn

class BaseEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(2, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2) # Output h, k
        )
    def predict(self, h):
        return self.head(h.mean(dim=0))
