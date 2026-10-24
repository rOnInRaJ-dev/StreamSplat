# Pure-PyTorch port of the forward pass of submodules/diff-gaussian-rasterization-orth,
# used when the CUDA extension is unavailable (e.g. macOS / MPS). Same math as
# cuda_rasterizer/forward.cu: orthographic EWA projection, 16x16 tile culling,
# front-to-back compositing with the same alpha clamps and early termination.
# Backward comes from autograd (means2D gets no gradient).

from typing import NamedTuple
import torch
import torch.nn as nn

BLOCK = 16


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


def quat_to_rot(q):
    # not normalized, matching computeCov3D
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def preprocess(means3D, opacities, scales, rotations, s: GaussianRasterizationSettings):
    """Per-Gaussian screen-space data, mirroring preprocessCUDA."""
    H, W = s.image_height, s.image_width
    vm = s.viewmatrix.to(means3D)  # stored transposed, so points are row vectors
    pm = s.projmatrix.to(means3D)
    p_hom = torch.cat([means3D, torch.ones_like(means3D[:, :1])], -1)
    p_view = p_hom @ vm
    p_proj = p_hom @ pm
    p_proj = p_proj[:, :2] / (p_proj[:, 3:] + 1e-7)

    R = quat_to_rot(rotations)
    M = R * (s.scale_modifier * scales)[:, None, :]
    cov3D = M @ M.transpose(1, 2)
    W3 = vm[:3, :3]
    cov = W3.T @ cov3D @ W3
    fx, fy = W / (2 * s.tanfovx), H / (2 * s.tanfovy)
    a = fx * fx * cov[:, 0, 0] + 0.3
    b = fx * fy * cov[:, 0, 1]
    c = fy * fy * cov[:, 1, 1] + 0.3
    det = a * c - b * b
    conic = torch.stack([c, -b, a], -1) / det[:, None]

    mid = 0.5 * (a + c)
    radius = torch.ceil(3 * torch.sqrt(mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))))
    xy = ((p_proj + 1) * torch.tensor([W, H], device=means3D.device) - 1) * 0.5

    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    hi = torch.tensor([gx, gy], device=means3D.device)
    r = radius.detach()[:, None]
    p = xy.detach().nan_to_num(0.0)
    rect_min = torch.minimum(torch.trunc((p - r) / BLOCK).clamp(min=0), hi).long()
    rect_max = torch.minimum(torch.trunc((p + r + BLOCK - 1) / BLOCK).clamp(min=0), hi).long()
    area = (rect_max - rect_min).prod(-1)
    valid = (p_view[:, 2] > 0) & (det != 0) & (area > 0) & torch.isfinite(radius)
    return dict(xy=xy, depth=p_view[:, 2], conic=conic, opacity=opacities[:, 0],
                radius=radius, rect_min=rect_min, rect_max=rect_max, valid=valid)


def bin_tiles(g, num_tiles_x, num_tiles):
    """Per-tile Gaussian lists sorted front to back: [num_tiles, K], padded with -1."""
    ids = torch.nonzero(g["valid"])[:, 0]
    ids = ids[torch.sort(g["depth"].detach()[ids], stable=True).indices]
    lo, hi = g["rect_min"][ids], g["rect_max"][ids]
    wx = hi[:, 0] - lo[:, 0]
    counts = wx * (hi[:, 1] - lo[:, 1])
    owner = torch.repeat_interleave(torch.arange(len(ids), device=ids.device), counts)
    local = torch.arange(len(owner), device=ids.device) - (torch.cumsum(counts, 0) - counts)[owner]
    tile = (lo[owner, 1] + local // wx[owner]) * num_tiles_x + lo[owner, 0] + local % wx[owner]
    # owner is already depth-ordered, so key = (tile, owner) is unique and sorts tile-major, depth-minor
    order = torch.sort(tile * len(ids) + owner).indices
    tile, gid = tile[order], ids[owner[order]]
    per_tile = torch.bincount(tile, minlength=num_tiles)
    slot = torch.arange(len(tile), device=ids.device) - (torch.cumsum(per_tile, 0) - per_tile)[tile]
    table = torch.full((num_tiles, int(per_tile.max()) if len(tile) else 0), -1, dtype=torch.long, device=ids.device)
    table[tile, slot] = gid
    return table


def rasterize(means3D, colors, opacities, scales, rotations, s: GaussianRasterizationSettings, chunk_elems=1 << 24):
    H, W = s.image_height, s.image_width
    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    g = preprocess(means3D, opacities, scales, rotations, s)
    radii = torch.where(g["valid"], g["radius"], 0).int()
    table = bin_tiles(g, gx, gx * gy)
    K = table.shape[1]
    bg = s.bg.to(means3D)

    # padded slot -1 points at an appended zero-opacity Gaussian
    xy = torch.cat([g["xy"], g["xy"].new_zeros(1, 2)])
    conic = torch.cat([g["conic"], g["conic"].new_zeros(1, 3)])
    opacity = torch.cat([g["opacity"], g["opacity"].new_zeros(1)])
    feats = torch.cat([colors, g["depth"][:, None]], -1)
    feats = torch.cat([feats, feats.new_zeros(1, 4)])

    pix = torch.stack(torch.meshgrid(torch.arange(BLOCK), torch.arange(BLOCK), indexing="ij")[::-1], -1)
    pix = pix.reshape(-1, 2).to(means3D)  # (x, y) inside a tile
    tile_xy = torch.stack(torch.meshgrid(torch.arange(gy), torch.arange(gx), indexing="ij")[::-1], -1)
    tile_xy = tile_xy.reshape(-1, 1, 2).to(means3D) * BLOCK

    out = []
    step = max(1, chunk_elems // max(1, BLOCK * BLOCK * K))
    for t0 in range(0, gx * gy, step):
        ids = table[t0:t0 + step]  # [T, K]
        d = xy[ids][:, None] - (tile_xy[t0:t0 + step] + pix)[:, :, None]  # [T, P, K, 2]
        con = conic[ids][:, None]
        power = -0.5 * (con[..., 0] * d[..., 0] ** 2 + con[..., 2] * d[..., 1] ** 2) - con[..., 1] * d[..., 0] * d[..., 1]
        alpha = torch.clamp(opacity[ids][:, None] * torch.exp(power), max=0.99)
        alpha = torch.where((power > 0) | (alpha < 1.0 / 255.0), 0.0, alpha)
        T_after = torch.cumprod(1 - alpha, -1)
        keep = T_after >= 1e-4  # first failing Gaussian marks the pixel done
        T_before = torch.cat([torch.ones_like(T_after[..., :1]), T_after[..., :-1]], -1)
        w = alpha * T_before * keep
        acc = w @ feats[ids]  # [T, P, 4]
        T_final = torch.where(keep, T_after, 1.0).amin(-1) if K else torch.ones_like(acc[..., 0])
        out.append(torch.cat([acc[..., :3] + T_final[..., None] * bg, acc[..., 3:], w.sum(-1, keepdim=True)], -1))

    img = torch.cat(out).reshape(gy, gx, BLOCK, BLOCK, 5).permute(4, 0, 2, 1, 3).reshape(5, gy * BLOCK, gx * BLOCK)
    img = img[:, :H, :W]
    return img[:3], radii, img[3:4], img[4:5]


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        if shs is not None or colors_precomp is None or cov3D_precomp is not None or scales is None or rotations is None:
            raise NotImplementedError("torch rasterizer only supports colors_precomp + scales/rotations")
        return rasterize(means3D, colors_precomp, opacities, scales, rotations, self.raster_settings)


if __name__ == "__main__":
    # Self-check against a per-Gaussian loop that follows renderCUDA literally.
    import math

    def reference(means3D, colors, opacities, scales, rotations, s):
        H, W = s.image_height, s.image_width
        g = preprocess(means3D, opacities, scales, rotations, s)
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        tx, ty = xs // BLOCK, ys // BLOCK
        T = torch.ones(H, W); C = torch.zeros(3, H, W); D = torch.zeros(H, W); A = torch.zeros(H, W)
        done = torch.zeros(H, W, dtype=torch.bool)
        for i in sorted(torch.nonzero(g["valid"])[:, 0].tolist(), key=lambda i: g["depth"][i].item()):
            (x0, y0), (x1, y1) = g["rect_min"][i].tolist(), g["rect_max"][i].tolist()
            in_rect = (tx >= x0) & (tx < x1) & (ty >= y0) & (ty < y1)
            dx, dy = g["xy"][i, 0] - xs, g["xy"][i, 1] - ys
            ca, cb, cc = g["conic"][i]
            power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy
            alpha = torch.clamp(g["opacity"][i] * torch.exp(power), max=0.99)
            live = in_rect & ~done & (power <= 0) & (alpha >= 1.0 / 255.0)
            test_T = T * (1 - alpha)
            done |= live & (test_T < 1e-4)
            live &= test_T >= 1e-4
            w = torch.where(live, alpha * T, 0.0)
            C += w * colors[i][:, None, None]; D += w * g["depth"][i]; A += w
            T = torch.where(live, test_T, T)
        return C + T * s.bg[:, None, None], D[None], A[None]

    # the fixed camera used by gaussian_renderer_dynamic: world-to-view (x, y, z) -> (x, -z, y), ortho proj
    w2c = torch.tensor([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    P = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, -0.2, -1], [0, 0, 0, 1]])
    torch.manual_seed(0)
    n = 300
    means = torch.stack([torch.rand(n) * 2.4 - 1.2, torch.rand(n) * 1.2 - 0.1, torch.rand(n) * 2.4 - 1.2], -1)
    means[:5, 1] = means[5:10, 1]  # depth ties
    rots = torch.randn(n, 4) * 0.5
    rots[:, 0] += 1
    scales = torch.rand(n, 3) * 0.08
    scales[:20] *= 5  # a few that span many tiles
    args = (means, torch.rand(n, 3), torch.rand(n, 1), scales, rots)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    for H, W in [(48, 80), (37, 53)]:
        s = GaussianRasterizationSettings(H, W, math.tan(math.pi / 4), math.tan(math.pi / 4), torch.tensor([0.5, 0.5, 0.5]),
                                          1.0, w2c.T.contiguous(), (P @ w2c).T.contiguous(), 0, torch.zeros(3), False, False)
        # closed form for identity rotation: pixel (((x+1)W-1)/2, ((1-z)H-1)/2), depth y, cov diag(fx^2 sx^2, fy^2 sz^2) + 0.3
        g = preprocess(means[:1], torch.ones(1, 1), torch.tensor([[0.01, 0.02, 0.03]]), torch.tensor([[1., 0, 0, 0]]), s)
        x, y, z = means[0].tolist()
        assert torch.allclose(g["xy"][0], torch.tensor([((x + 1) * W - 1) / 2, ((1 - z) * H - 1) / 2]))
        assert torch.allclose(g["depth"][0], torch.tensor(y))
        assert torch.allclose(g["conic"][0], torch.tensor([1 / ((W / 2 * 0.01) ** 2 + 0.3), 0, 1 / ((H / 2 * 0.03) ** 2 + 0.3)]))
        ref = reference(*args, s)
        for dev in devices:
            sd = s._replace(bg=s.bg.to(dev), viewmatrix=s.viewmatrix.to(dev), projmatrix=s.projmatrix.to(dev))
            img, radii, depth, alpha = rasterize(*[a.to(dev) for a in args], sd, chunk_elems=4096)
            for name, got, want in zip(["color", "depth", "alpha"], [img, depth, alpha], ref):
                err = (got.cpu() - want).abs().max().item()
                assert err < 1e-4, f"{dev} {H}x{W} {name} max err {err}"
            print(f"ok {dev} {H}x{W}")

    # gradients flow
    m = args[0].clone().requires_grad_()
    rasterize(m, *args[1:], s)[0].sum().backward()
    assert m.grad.abs().sum() > 0
    print("all good")
