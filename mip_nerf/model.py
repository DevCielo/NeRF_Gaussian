import torch, torch.nn as nn
from hashencoding import HashGridEncoder
from ray_utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map
from pose_utils import to8b


class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super().__init__()
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is None:
            return torch.sin(x_enc)
        y_enc = (y[..., None, :] * self.scales[:, None] ** 2).reshape(shape)
        y_enc = torch.cat((y_enc, y_enc), -1)
        x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
        y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
        return x_ret, y_ret


def _xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class MipNeRF(nn.Module):
    def __init__(
        self,
        use_viewdirs=True,
        randomized=False,
        ray_shape="cone",
        white_bkgd=True,
        num_levels=2,
        num_samples=128,
        hidden=256,
        density_noise=1,
        density_bias=-1,
        rgb_padding=0.001,
        resample_padding=0.01,
        min_deg=0,
        max_deg=16,
        viewdirs_min_deg=0,
        viewdirs_max_deg=4,
        device=torch.device("cpu"),
        return_raw=False,
        use_hash_encoding=False,
        hash_levels=16,
        hash_features=2,
    ):
        super().__init__()
        self.use_hash = use_hash_encoding
        self.use_viewdirs = use_viewdirs
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.density_bias = density_bias
        self.return_raw = return_raw
        self.device = device

        if self.use_hash:
            self.encoder = HashGridEncoder(num_levels=hash_levels, features_per_level=hash_features)
            density_in = hash_levels * hash_features
        else:
            self.encoder = PositionalEncoding(min_deg, max_deg)
            density_in = (max_deg - min_deg) * 3 * 2

        self.density_net = nn.Sequential(
            nn.Linear(density_in, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Linear(hidden, 1)

        rgb_in_hidden = hidden
        if self.use_viewdirs:
            self.viewdir_enc = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            rgb_in_hidden += 3 + (viewdirs_max_deg - viewdirs_min_deg) * 3 * 2
        self.rgb_net = nn.Sequential(nn.Linear(rgb_in_hidden, hidden), nn.ReLU(True), nn.Linear(hidden, hidden), nn.ReLU(True))
        self.final_rgb = nn.Sequential(nn.Linear(hidden, 3), nn.Sigmoid())
        self.density_act = nn.Softplus()
        self.apply(_xavier_init)
        self.to(device)

    def _encode_positions(self, mean, var):
        if self.use_hash:
            return self.encoder((mean - mean.min()) / (mean.max() - mean.min() + 1e-10))
        return self.encoder(mean, var)[0]

    def forward(self, rays):
        comp_rgbs = []
        distances = []
        accs = []
        for l in range(self.num_levels):
            if l == 0:
                t_vals, (mean, var) = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized=self.randomized,
                    lindisp=False,
                    ray_shape=self.ray_shape,
                )
            else:
                t_vals, (mean, var) = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_vals.to(rays.origins.device),
                    weights.to(rays.origins.device),
                    randomized=self.randomized,
                    stop_grad=True,
                    resample_padding=self.resample_padding,
                    ray_shape=self.ray_shape,
                )

            enc = self._encode_positions(mean, var)
            batch, n_samples, feat_dim = enc.shape
            h = enc.reshape(batch * n_samples, feat_dim)

            h = self.density_net(h)
            raw_density = self.final_density(h).view(-1, self.num_samples, 1)

            if self.use_viewdirs:
                v_enc = self.viewdir_enc(rays.viewdirs.to(self.device))
                v = torch.cat((v_enc, rays.viewdirs.to(self.device)), -1)
                v = v.repeat_interleave(self.num_samples, 0)
                h_rgb = torch.cat((h, v), -1)
            else:
                h_rgb = h

            h_rgb = self.rgb_net(h_rgb)
            raw_rgb = self.final_rgb(h_rgb).view(-1, self.num_samples, 3)

            if self.randomized and self.density_noise:
                raw_density = raw_density + self.density_noise * torch.rand_like(raw_density)

            rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_act(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights, _ = volumetric_rendering(
                rgb, density, t_vals, rays.directions.to(rgb.device), self.white_bkgd
            )

            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)

        if self.return_raw:
            raws = torch.cat((raw_rgb.detach(), density.detach()), -1).cpu()
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws

        return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

    def render_image(self, rays, h, w, chunks=8192):
        n = rays[0].shape[0]
        rgbs, ds, accs = [], [], []
        with torch.no_grad():
            for i in range(0, n, chunks):
                chunk = namedtuple_map(lambda r: r[i : i + chunks].to(self.device), rays)
                rgb, d, a = self(chunk)
                rgbs.append(rgb[-1].cpu())
                ds.append(d[-1].cpu())
                accs.append(a[-1].cpu())
        img = to8b(torch.cat(rgbs).view(h, w, 3).numpy())
        dist = torch.cat(ds).view(h, w).numpy()
        acc = torch.cat(accs).view(h, w).numpy()
        return img, dist, acc

    def train(self, mode=True):
        self.randomized = mode and self.randomized
        return super().train(mode)

    def eval(self):
        self.randomized = False
        return super().eval()
