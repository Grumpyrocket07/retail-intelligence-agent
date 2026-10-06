"""The PyTorch neural network for churn (kept separate so the other models run without PyTorch installed)."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(xtr, ytr, xva, yva, seed: int, max_epochs: int = 150, patience: int = 15) -> MLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr.astype(np.float32))
    xv = torch.from_numpy(xva)
    best_ap, best_state, wait = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(yva, torch.sigmoid(model(xv)).numpy())
        if ap > best_ap:
            best_ap, wait = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def mlp_proba(models, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(x)
        return np.mean([torch.sigmoid(m(xt)).numpy() for m in models], axis=0)


def build_mlp(n_in: int) -> MLP:
    """Rebuild an empty network of the right shape, e.g. before loading saved weights."""
    return MLP(n_in)
