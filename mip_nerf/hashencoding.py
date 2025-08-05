# hashcoding.py
import math, torch, torch.nn as nn, torch.nn.functional as F

_PRIMES = (1_000_003, 1_000_033, 1_000_087)


def _hash(coords, size_mask):
    return ((coords[:, 0] * _PRIMES[0]) ^ (coords[:, 1] * _PRIMES[1]) ^ (coords[:, 2] * _PRIMES[2])) & size_mask


class HashGridEncoder(nn.Module):
    def __init__(
        self,
        num_levels=16,
        features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=1.5,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.hashmap_size = 1 << log2_hashmap_size
        self.size_mask = self.hashmap_size - 1
        tables = [nn.Embedding(self.hashmap_size, features_per_level) for _ in range(num_levels)]
        self.tables = nn.ModuleList(tables)
        for t in self.tables:
            nn.init.uniform_(t.weight, -1e-4, 1e-4)

    def forward(self, x):  # x ∈ [0,1]^3  shape (N,3)
        x = x.clamp(0, 1)
        outputs = []
        for lvl in range(self.num_levels):
            res = self.base_resolution * (self.per_level_scale ** lvl)
            res_i = int(math.floor(res))
            pos = x * res_i
            idx0 = pos.floor().int()
            frac = pos - idx0.float()
            c = []
            w = []
            for xi in (0, 1):
                for yi in (0, 1):
                    for zi in (0, 1):
                        offs = torch.stack((idx0[:, 0] + xi, idx0[:, 1] + yi, idx0[:, 2] + zi), -1)
                        h = _hash(offs, self.size_mask)
                        c.append(self.tables[lvl](h))
                        w.append(((1 - xi) + (-1 + 2 * xi) * frac[:, 0]) *
                                 ((1 - yi) + (-1 + 2 * yi) * frac[:, 1]) *
                                 ((1 - zi) + (-1 + 2 * zi) * frac[:, 2]))
            feats = sum(wi.unsqueeze(-1) * ci for wi, ci in zip(w, c))
            outputs.append(feats)
        return torch.cat(outputs, -1)
