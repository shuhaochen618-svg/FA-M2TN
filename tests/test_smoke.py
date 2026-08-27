import unittest

import torch

from pg_m2tn.data.masking_engine import MaskedBatteryDataset
from pg_m2tn.models.loss import FixedWeightedLoss
from pg_m2tn.models.pg_m2tn import PGM2TN, count_parameters
from pg_m2tn.protocol import VARIANTS


class SingleSampleDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "features": torch.linspace(-1.0, 1.0, 1024).reshape(512, 2),
            "soh": torch.tensor(0.9),
            "vdr": torch.tensor(1.1),
            "cycle_idx": 0,
            "cycle_fraction": 0.0,
            "cell_id": "CALCE_TEST",
            "dataset": "CALCE",
        }


class PGM2TNSmokeTest(unittest.TestCase):
    def test_model_shapes_and_parameter_count(self):
        model = PGM2TN(hidden_dim=128, num_layers=2)
        reconstruction, soh, vdr = model(torch.randn(2, 512, 2))
        self.assertEqual(tuple(reconstruction.shape), (2, 512, 2))
        self.assertEqual(tuple(soh.shape), (2, 1))
        self.assertEqual(tuple(vdr.shape), (2, 1))
        self.assertEqual(count_parameters(model), 630149)

    def test_fixed_mask_is_reproducible(self):
        first = MaskedBatteryDataset(
            SingleSampleDataset(), fixed_mask_ratio=0.5, seed=20260821
        )[0]
        second = MaskedBatteryDataset(
            SingleSampleDataset(), fixed_mask_ratio=0.5, seed=20260821
        )[0]
        self.assertTrue(torch.equal(first["binary_mask"], second["binary_mask"]))
        self.assertEqual(int(first["binary_mask"].sum()), 256)

    def test_full_loss_uses_published_fixed_weights(self):
        configuration = VARIANTS["full"]
        criterion = FixedWeightedLoss(
            configuration["soh_weight"],
            configuration["vdr_weight"],
            configuration["lambda_mae"],
        )
        batch = {
            "x_full": torch.ones(1, 4, 2),
            "binary_mask": torch.tensor([[True, True, False, False]]),
            "soh": torch.tensor([1.0]),
            "vdr": torch.tensor([1.0]),
        }
        total, parts = criterion(
            batch,
            torch.zeros(1, 4, 2),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
        self.assertAlmostEqual(parts["soh"], 1.0)
        self.assertAlmostEqual(parts["vdr"], 1.0)
        self.assertAlmostEqual(parts["mae"], 1.0)
        self.assertAlmostEqual(float(total), 1.1)


if __name__ == "__main__":
    unittest.main()
