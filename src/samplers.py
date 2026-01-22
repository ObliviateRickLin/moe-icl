import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False, device='cpu'):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims, device=device))
    t = U @ torch.diag(eigenvalues.to(device)) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device='cpu'):
        """
        Sample xs on the specified device (CPU or GPU).
        
        Args:
            n_points: number of points per sample
            b_size: batch size
            n_dims_truncated: dimensions to truncate (set to 0)
            seeds: optional seeds for reproducibility
            device: 'cpu' or 'cuda' - where to generate the data
        """
        if seeds is None:
            # Direct generation on target device (fast path)
            xs_b = torch.randn(b_size, n_points, self.n_dims, device=device)
        else:
            # Seeded generation - must use CPU generator then transfer
            # (CUDA generators have different behavior)
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
            xs_b = xs_b.to(device)
        if self.scale is not None:
            scale = self.scale.to(device) if isinstance(self.scale, torch.Tensor) else self.scale
            xs_b = xs_b @ scale
        if self.bias is not None:
            bias = self.bias.to(device) if isinstance(self.bias, torch.Tensor) else self.bias
            xs_b = xs_b + bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
