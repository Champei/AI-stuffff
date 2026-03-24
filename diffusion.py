import torch

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1. - betas
alpha_bar = torch.cumprod(alphas, dim=0)
def q_sample(x0, t, noise):
    sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_1_ab = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]
    return sqrt_ab * x0 + sqrt_1_ab * noise
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x, t):
        return self.net(x)
    model = SimpleUNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for x0 in dataloader:
    t = torch.randint(0, T, (x0.size(0),))
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)

    pred_noise = model(xt, t)
    loss = ((noise - pred_noise) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    @torch.no_grad()
def sample():
    x = torch.randn(1, 3, 64, 64)
    for t in reversed(range(T)):
        eps = model(x, torch.tensor([t]))
        x = (1 / torch.sqrt(alphas[t])) * (
            x - betas[t] / torch.sqrt(1 - alpha_bar[t]) * eps
        )
        if t > 0:
            x += torch.sqrt(betas[t]) * torch.randn_like(x)
    return x