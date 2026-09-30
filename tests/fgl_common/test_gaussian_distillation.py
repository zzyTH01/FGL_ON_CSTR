"""Gaussian distribution distillation: model, losses, training smoke test."""
import math
import pytest
import torch
import numpy as np

from fgl_common.distillation import gaussian_nll, gaussian_kl
from fgl_common.models import RNNGaussian


class TestGaussianNLL:
    def test_perfect_prediction_with_tight_sigma_below_zero(self):
        mu = torch.tensor([1.0, 2.0])
        log_sigma = torch.full((2,), math.log(0.1))  # σ=0.1
        targets = torch.tensor([1.0, 2.0])
        nll = gaussian_nll(mu, log_sigma, targets)
        assert nll.item() < 0  # 0.5*(0 + 2*log(0.1) + log(2π)) ≈ -1.38

    def test_wrong_prediction_higher_loss(self):
        mu = torch.tensor([1.0])
        log_sigma = torch.tensor([0.0])
        close = gaussian_nll(mu, log_sigma, torch.tensor([1.1]))
        far = gaussian_nll(mu, log_sigma, torch.tensor([3.0]))
        assert far > close

    def test_zero_mean_unit_variance_baseline(self):
        mu = torch.zeros(1)
        log_sigma = torch.zeros(1)
        targets = torch.zeros(1)
        nll = gaussian_nll(mu, log_sigma, targets)
        assert abs(nll.item() - 0.5 * math.log(2 * math.pi)) < 1e-5

    def test_scalar_output(self):
        out = gaussian_nll(torch.zeros(4), torch.zeros(4), torch.ones(4))
        assert out.dim() == 0


class TestGaussianKL:
    def test_identical_distributions_zero_kl(self):
        mu = torch.zeros(8)
        log_sigma = torch.zeros(8)
        kl = gaussian_kl(mu, log_sigma, mu, log_sigma, temperature=1.0, alpha=0.0)
        assert abs(kl.item()) < 1e-6

    def test_different_means_positive_kl(self):
        mu_t = torch.zeros(8)
        mu_s = torch.ones(8)
        log_sigma = torch.zeros(8)
        kl = gaussian_kl(mu_t, log_sigma, mu_s, log_sigma, temperature=1.0, alpha=0.0)
        assert kl.item() > 0

    def test_alpha_one_gives_zero(self):
        kl = gaussian_kl(torch.zeros(4), torch.zeros(4),
                         torch.ones(4), torch.zeros(4),
                         temperature=4.0, alpha=1.0)
        assert abs(kl.item()) < 1e-6

    def test_temperature_changes_kl(self):
        mu_t, mu_s = torch.zeros(4), torch.tensor([0.5] * 4)
        ls = torch.zeros(4)
        kl_low = gaussian_kl(mu_t, ls, mu_s, ls, temperature=1.0, alpha=0.0)
        kl_high = gaussian_kl(mu_t, ls, mu_s, ls, temperature=4.0, alpha=0.0)
        assert kl_high != kl_low


class TestRNNGaussian:
    def test_output_shapes(self):
        model = RNNGaussian(input_size=20, hidden_size=32, num_layers=2)
        x = torch.randn(4, 1, 20)  # (batch, seq_len=1, features=lookback)
        mu, log_sigma = model(x)
        assert mu.shape == (4,)
        assert log_sigma.shape == (4,)

    def test_forward_pass_differentiable(self):
        model = RNNGaussian(input_size=10, hidden_size=16)
        x = torch.randn(2, 1, 10)
        mu, log_sigma = model(x)
        loss = gaussian_nll(mu, log_sigma, torch.ones(2))
        loss.backward()


class TestRunFGLGaussian:
    @pytest.fixture
    def toy_data(self):
        t = np.linspace(0, 10 * np.pi, 200)
        np.random.seed(42)
        vals = (np.sin(t) + 1) / 2 + 0.01 * np.random.randn(len(t))
        return [(float(v), float(v)) for v in vals]

    def test_returns_expected_keys(self, toy_data):
        from fgl_common.training import run_fgl_gaussian
        result = run_fgl_gaussian(
            toy_data, lookback_window=10, forecasting_horizon=3,
            alpha=0.5, temperature=2.0, epochs=3, seed=0, verbose=False,
        )
        for key in ("teacher", "baseline", "student", "improvement",
                     "student_rmse", "baseline_rmse"):
            assert key in result

    def test_physical_mse_is_finite_positive(self, toy_data):
        from fgl_common.training import run_fgl_gaussian
        result = run_fgl_gaussian(
            toy_data, lookback_window=10, forecasting_horizon=3,
            alpha=0.5, temperature=2.0, epochs=3, seed=0, verbose=False,
        )
        assert result["student"] >= 0
        assert math.isfinite(result["student"])
