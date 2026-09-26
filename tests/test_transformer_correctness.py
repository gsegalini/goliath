import unittest
from types import SimpleNamespace

import torch

from src.diffusion_process import GaussianDiffusionProcess, MaskedDiffusionProcess
from src.model import (
    FTTransformerDiffusion,
    build_diffusion_model,
    diffusion_model_config,
)
from src.train_diffusion import _normalize_attack_latent_args


def _args(**overrides):
    values = {
        "attack_lr": 0.06,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "known_mask_amount": 0,
        "known_noise_amount": 0,
        "known_t_amount": 0,
        "no_mask": False,
        "no_noise": False,
        "no_prior": False,
        "no_t": False,
        "optim_iters": 20,
        "prior_cooldown_start": None,
        "prior_peak": None,
        "random_latents": "",
        "simple_attack": False,
        "t_init": "midpoint",
        "timesteps": 50,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _masked_process():
    return MaskedDiffusionProcess(
        _args(),
        torch.device("cpu"),
        feat_info={"cont": 2, "cat": [2, 3]},
    )


class TransformerCorrectnessTests(unittest.TestCase):
    def test_transformer_class_defaults_match_experiment_defaults(self):
        model = FTTransformerDiffusion(d_in=5, d_out=5)
        self.assertEqual(model.d_model, 256)
        self.assertEqual(model.nhead, 8)
        self.assertEqual(model.nlayers, 4)
        self.assertEqual(model.dim_feedforward, 512)

    def test_transformer_model_config_round_trip(self):
        torch.manual_seed(0)
        model = FTTransformerDiffusion(
            d_in=9,
            d_out=7,
            d_model=8,
            nhead=2,
            nlayers=1,
            dim_feedforward=16,
            feat_init_std=0.5,
            head_init_std=0.01,
        )
        rebuilt = build_diffusion_model(diffusion_model_config(model))
        rebuilt.load_state_dict(model.state_dict())

        x = torch.randn(3, 9)
        t = torch.tensor([0.0, 12.0, 49.0])
        torch.testing.assert_close(rebuilt(x, t), model(x, t))

    def test_masked_layout_keeps_data_features_before_mask_tokens(self):
        process = _masked_process()
        self.assertEqual(process.cat_feature_indices.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(process.cat_mask_indices.tolist(), [5, 6])

        x_clean = torch.tensor([[0.25, -0.5, 1.0, 0.0, 0.0, 1.0, 0.0]])
        latents = {
            "t": torch.tensor([0]),
            "noise": torch.zeros(1, 2),
            "mask": torch.tensor([[1.0, 0.0]]),
        }
        x_noised = process.forward_diffuse(x_clean, latents)

        # First categorical group is masked, second is retained, and both indicators are last.
        torch.testing.assert_close(x_noised[0, 2:7], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
        torch.testing.assert_close(x_noised[0, 7:], torch.tensor([1.0, 0.0]))

    def test_masked_training_and_attack_losses_are_finite(self):
        torch.manual_seed(0)
        process = _masked_process()
        model = FTTransformerDiffusion(
            d_in=9,
            d_out=7,
            d_model=8,
            nhead=2,
            nlayers=1,
            dim_feedforward=16,
        )
        x_clean = torch.tensor(
            [
                [0.25, -0.5, 1.0, 0.0, 0.0, 1.0, 0.0],
                [-0.1, 0.4, 0.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )
        latents = process.sample_latents(x_clean, torch.device("cpu"))
        x_noised = process.forward_diffuse(x_clean, latents)
        train_loss = process.compute_training_loss(model, x_noised, latents)
        self.assertTrue(torch.isfinite(train_loss))

        known = process.mask_known_latents(latents, _args(), len(x_clean), torch.device("cpu"))
        dummy_latents, _, _ = process.init_dummy_latents(known, _args())
        process.set_attack_iteration(5)
        attack_noised = process.forward_diffuse_attack(x_clean, dummy_latents, len(x_clean))
        attack_loss = process.compute_attack_inner_loss(model, attack_noised, dummy_latents)
        self.assertTrue(torch.isfinite(attack_loss))

        predicted_grads = torch.autograd.grad(attack_loss, model.parameters(), create_graph=True)
        meta_loss = sum(gradient.square().mean() for gradient in predicted_grads)
        dummy_grad = torch.autograd.grad(meta_loss, dummy_latents["mask_logits"])[0]
        self.assertTrue(torch.isfinite(dummy_grad).all())

    def test_mask_prior_updates_timestep_and_mask_logits(self):
        process = _masked_process()
        process.init_dummy_latents(
            {
                "t": torch.full((2,), float("nan")),
                "mask": torch.full((2, 2), float("nan")),
                "noise": torch.full((2, 2), float("nan")),
            },
            _args(),
        )
        process.set_attack_iteration(10)
        process._cached_alpha_bar = None

        timestep = torch.tensor([8.0, 35.0], requires_grad=True)
        mask_logits = torch.tensor([[-2.0, 1.0], [0.5, 3.0]], requires_grad=True)
        prior = process.compute_latent_prior_loss(
            {"t": timestep, "mask_logits": mask_logits},
        )
        timestep_grad, mask_grad = torch.autograd.grad(prior, (timestep, mask_logits))

        self.assertTrue(torch.isfinite(prior))
        self.assertGreater(timestep_grad.abs().sum(), 0)
        self.assertGreater(mask_grad.abs().sum(), 0)

    def test_known_t_and_noise_do_not_expose_unknown_mask(self):
        args = _args(
            diffusion_type="masked",
            known_t_amount=100,
            known_noise_amount=100,
            known_mask_amount=0,
        )
        _normalize_attack_latent_args(args)

        self.assertFalse(args.simple_attack)
        self.assertEqual(args.known_mask_amount, 0)

        args.known_mask_amount = 100
        _normalize_attack_latent_args(args)
        self.assertTrue(args.simple_attack)

    def test_transformer_gaussian_attack_uses_full_noise_learning_rate(self):
        ft_args = _args(backbone="ft-transformer")
        mlp_args = _args(backbone="mlp")
        ft_process = GaussianDiffusionProcess(ft_args, torch.device("cpu"))
        mlp_process = GaussianDiffusionProcess(mlp_args, torch.device("cpu"))
        override_process = GaussianDiffusionProcess(
            _args(backbone="ft-transformer", noise_lr_mult=0.25),
            torch.device("cpu"),
        )

        self.assertEqual(ft_process.noise_lr_mult, 1.0)
        self.assertEqual(mlp_process.noise_lr_mult, 0.1)
        self.assertEqual(override_process.noise_lr_mult, 0.25)


if __name__ == "__main__":
    unittest.main()
