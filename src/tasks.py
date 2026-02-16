import math

import torch
import pdb

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, device='cpu'):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        self.device = device
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "noisy_quadratic_regression": NoisyQuadraticRegression,
        "noisy_quadratic_regression_mix4": NoisyQuadraticRegressionMix4,
        # Backward-compatible typo aliases.
        "noise_quatradic_regression": NoisyQuadraticRegression,
        "noisy_quatradic_regression": NoisyQuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "noisy_relu_2nn_regression": NoisyRelu2nnRegression,
        "decision_tree": DecisionTree,
        "noisy_decision_tree": NoisyDecisionTree,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, normalize_w=False, device='cpu'):
        """
        scale: a constant by which to scale the randomly sampled weights.
        device: 'cpu' or 'cuda' - where to generate weights.
        """
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds, device)
        self.scale = scale
        self.normalize_w = normalize_w

        if pool_dict is None and seeds is None:
            # Direct generation on target device (fast path)
            self.w_b = torch.randn(self.b_size, self.n_dims, 1, device=device)
        elif seeds is not None:
            # Seeded generation - use CPU then transfer for reproducibility
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
            self.w_b = self.w_b.to(device)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices].to(device)

    def evaluate(self, xs_b):
        # w_b is already on the correct device from __init__
        w_b = self.w_b.to(xs_b.device)  # ensure same device as xs_b
        scale = self.scale / math.sqrt(self.n_dims) if self.normalize_w else self.scale
        ys_b = scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
        normalize_w=False,
        device='cpu',
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, device=device
        )
        self.sparsity = sparsity
        self.normalize_w = normalize_w
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        # Apply sparsity mask (on CPU for reproducibility, then ensure device)
        w_b_cpu = self.w_b.cpu()
        for i, w in enumerate(w_b_cpu):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0
        self.w_b = w_b_cpu.to(device)

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        scale = self.scale / math.sqrt(self.sparsity) if self.normalize_w else self.scale
        ys_b = scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def __init__(
            self,
            n_dims,
            batch_size,
            pool_dict=None,
            seeds=None,
            scale=1,
            logistic=False,
            normalize_w=False,
            device='cpu',
    ):
        super(LinearClassification, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, normalize_w=normalize_w, device=device
        )
        self.logistic = logistic

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        if self.logistic:
            ps_b = sigmoid(ys_b)
            # Generate random tensor on same device as ps_b
            ys_b = (torch.rand(ps_b.size(), device=ps_b.device) < ps_b).float()
            return ys_b * 2 - 1
        else:
            # this is equivalent to logistic regression with infinite coefficient norm
            return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
        normalize_w=False,
        device='cpu',
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, normalize_w=normalize_w, device=device
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        # torch.randn_like generates on same device as ys_b
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        scale = self.scale / math.sqrt(self.n_dims) if self.normalize_w else self.scale
        ys_b_quad = scale * ys_b_quad
        return ys_b_quad


class NoisyQuadraticRegression(QuadraticRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
        normalize_w=False,
        device='cpu',
    ):
        """
        noise_std: standard deviation of additive Gaussian noise.
        """
        super(NoisyQuadraticRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, normalize_w=normalize_w, device=device
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy


class NoisyQuadraticRegressionMix4(NoisyQuadraticRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_stds=(0.1, 0.25, 0.5, 1.0),
        renormalize_ys=False,
        normalize_w=False,
        device='cpu',
    ):
        """
        4-level mixed-noise quadratic regression.
        Each batch element draws one sigma from `noise_stds` and applies it to all
        points in that element.
        """
        super(NoisyQuadraticRegressionMix4, self).__init__(
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            seeds=seeds,
            scale=scale,
            noise_std=0.0,
            renormalize_ys=renormalize_ys,
            normalize_w=normalize_w,
            device=device,
        )
        if len(noise_stds) != 4:
            raise ValueError("noise_stds must contain exactly 4 noise levels.")
        self.noise_stds = torch.tensor(noise_stds, dtype=torch.float32, device=device)

    def evaluate(self, xs_b):
        ys_b = QuadraticRegression.evaluate(self, xs_b)
        bsz = ys_b.shape[0]
        idx = torch.randint(
            low=0,
            high=4,
            size=(bsz,),
            device=ys_b.device,
        )
        sigmas = self.noise_stds.to(ys_b.device)[idx].view(bsz, 1)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * sigmas
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=4,
        normalize_w=False,
        device='cpu',
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds, device)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size
        self.normalize_w = normalize_w

        if pool_dict is None and seeds is None:
            # Direct generation on target device (fast path)
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size, device=device)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1, device=device)
        elif seeds is not None:
            # Seeded generation - use CPU then transfer
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
            self.W1 = self.W1.to(device)
            self.W2 = self.W2.to(device)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices].to(device)
            self.W2 = pool_dict["W2"][indices].to(device)

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        xs_eff = xs_b / math.sqrt(self.n_dims) if self.normalize_w else xs_b
        ys_b_nn = (torch.nn.functional.relu(xs_eff @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class NoisyRelu2nnRegression(Relu2nnRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=4,
        noise_std=0,
        renormalize_ys=False,
        normalize_w=False,
        device='cpu',
    ):
        """
        noise_std: standard deviation of additive Gaussian noise.
        """
        super(NoisyRelu2nnRegression, self).__init__(
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            seeds=seeds,
            scale=scale,
            hidden_layer_size=hidden_layer_size,
            normalize_w=normalize_w,
            device=device,
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4, device='cpu'):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds, device)
        self.depth = depth

        if pool_dict is None and seeds is None:
            # Direct generation on target device (fast path)
            # Note: randint on CUDA requires int64 dtype workaround
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1), device=device
            )
            self.target_tensor = torch.randn(self.dt_tensor.shape, device=device)
        elif seeds is not None:
            # Seeded generation - use CPU then transfer
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1, dtype=torch.long)
            self.target_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=(2 ** (depth + 1) - 1,),
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
            self.dt_tensor = self.dt_tensor.to(device)
            self.target_tensor = self.target_tensor.to(device)
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0], device=xs_b.device), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class NoisyDecisionTree(DecisionTree):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        depth=4,
        noise_std=0,
        renormalize_ys=False,
        device='cpu',
    ):
        """
        noise_std: standard deviation of additive Gaussian noise.
        """
        super(NoisyDecisionTree, self).__init__(
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            seeds=seeds,
            depth=depth,
            device=device,
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy
