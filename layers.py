import torch
from torch import nn

class LatentLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # layer for mean:
        self.w_mu = nn.Linear(in_dim, out_dim)
        # layer for variance:
        self.w_p = nn.Linear(in_dim, out_dim)

    def reparameterize(self, mu, p):
        # variance log(1 + exp(p))◦ eps:
        sigma = F.softplus(p)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.w_mu(x)
        p = self.w_p(x)
        z = self.reparameterize(mu, p)
        return (z, mu, p)