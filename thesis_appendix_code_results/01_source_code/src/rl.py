from __future__ import annotations
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random):
        self.buf = deque(maxlen=capacity)
        self.rng = rng

    def add(self, s, a, r, sp, done):
        self.buf.append((s, a, r, sp, done))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buf, k=min(batch_size, len(self.buf)))
        s, a, r, sp, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(sp), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(128,128)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def ddqn_update(q, q_tgt, opt, batch, gamma: float, device):
    s, a, r, sp, done = batch
    s  = torch.tensor(s, dtype=torch.float32, device=device)
    sp = torch.tensor(sp, dtype=torch.float32, device=device)
    a  = torch.tensor(a, dtype=torch.long, device=device)
    r  = torch.tensor(r, dtype=torch.float32, device=device)
    done = torch.tensor(done, dtype=torch.float32, device=device)

    with torch.no_grad():
        ap = torch.argmax(q(sp), dim=1)
        q_next = q_tgt(sp).gather(1, ap.view(-1,1)).squeeze(1)
        y = r + gamma * (1.0 - done) * q_next

    q_sa = q(s).gather(1, a.view(-1,1)).squeeze(1)
    loss = torch.mean((y - q_sa)**2)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
    opt.step()
    return float(loss.item())
