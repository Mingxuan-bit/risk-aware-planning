"""Scientific regression checks for disk probability and shared MC semantics."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_validation import (
    Disk,
    Gaussian2D,
    GaussianMixture2D,
    Rectangle,
    covering_rectangle,
    disk_reference,
    independent_union_probability,
    shared_space_mc,
)


class DiskProbabilityTests(unittest.TestCase):
    def test_centered_isotropic_gaussian_against_analytic_mass(self) -> None:
        for sigma, radius in ((0.3, 0.4), (1.0, 0.1), (0.1, 1.0), (2.0, 2.0)):
            gaussian = Gaussian2D((0.2, -0.3), sigma)
            disk = Disk(gaussian.mean, radius)
            exact = -math.expm1(-(radius * radius) / (2.0 * sigma * sigma))
            self.assertAlmostEqual(disk_reference(gaussian, disk), exact, places=11)

    def test_offset_integral_convergence_and_translation(self) -> None:
        gaussian = Gaussian2D((0.8, -0.2), 0.3)
        disk = Disk((0.0, 0.0), 0.5)
        coarse = disk_reference(gaussian, disk, panels=256)
        fine = disk_reference(gaussian, disk, panels=512)
        translated = disk_reference(
            Gaussian2D((10.8, -5.2), 0.3), Disk((10.0, -5.0), 0.5)
        )
        self.assertGreater(fine, 0.0)
        self.assertLess(fine, 1.0)
        self.assertAlmostEqual(coarse, fine, places=10)
        self.assertAlmostEqual(fine, translated, places=12)

    def test_gmm_weighted_integral_and_duplicate_modes(self) -> None:
        first = Gaussian2D((0.0, 0.0), 0.4)
        second = Gaussian2D((1.0, 0.2), 0.7)
        disk = Disk((0.2, 0.0), 0.3)
        mixture = GaussianMixture2D(((0.25, first), (0.75, second)))
        expected = 0.25 * disk_reference(first, disk) + 0.75 * disk_reference(second, disk)
        self.assertAlmostEqual(disk_reference(mixture, disk), expected, places=14)
        duplicate = GaussianMixture2D(((0.3, first), (0.7, first)))
        self.assertAlmostEqual(disk_reference(duplicate, disk), disk_reference(first, disk), places=14)
        self.assertAlmostEqual(duplicate.density((0.3, 0.2)), first.density((0.3, 0.2)), places=14)
        with self.assertRaises(ValueError):
            GaussianMixture2D(((0.4, first), (0.4, second)))

    def test_independent_event_union_and_reject_density_inputs(self) -> None:
        self.assertAlmostEqual(independent_union_probability((0.03, 0.04)), 0.0688, places=14)
        self.assertEqual(independent_union_probability(()), 0.0)
        self.assertEqual(independent_union_probability((1.0, 0.02)), 1.0)
        with self.assertRaises(ValueError):
            independent_union_probability((0.02, 1.4))
        with self.assertRaises(ValueError):
            independent_union_probability((0.02, -0.1))

    def test_complete_disk_geometry_is_required(self) -> None:
        disks = (Disk((-1.0, 2.0), 0.3), Disk((3.0, -2.0), 0.7))
        bounds = covering_rectangle(disks)
        self.assertTrue(all(bounds.covers(disk) for disk in disks))
        self.assertAlmostEqual(bounds.xmin, -1.3)
        self.assertAlmostEqual(bounds.xmax, 3.7)
        self.assertAlmostEqual(bounds.ymin, -2.7)
        self.assertAlmostEqual(bounds.ymax, 2.3)
        with self.assertRaises(ValueError):
            shared_space_mc(
                (Gaussian2D((0.0, 0.0), 1.0),),
                (Disk((0.0, 0.0), 1.0),),
                n_samples=10,
                seed=0,
                bounds=Rectangle(-0.5, 0.5, -0.5, 0.5),
            )

    def test_shared_mc_matches_reference_and_is_seed_reproducible(self) -> None:
        distributions = (Gaussian2D((0.0, 0.0), 0.4), Gaussian2D((0.7, 0.0), 0.6))
        disks = (Disk((0.2, 0.1), 0.3), Disk((0.5, -0.1), 0.25))
        result = shared_space_mc(distributions, disks, n_samples=20000, seed=721)
        repeated = shared_space_mc(distributions, disks, n_samples=20000, seed=721)
        self.assertEqual(result, repeated)
        for disk, estimate in zip(disks, result.disks):
            self.assertGreater(estimate.hit_count, 0)
            self.assertIsNotNone(estimate.conditional_probabilities)
            reference = tuple(disk_reference(distribution, disk) for distribution in distributions)
            for calculated, exact in zip(estimate.rectangle_probabilities, reference):
                self.assertAlmostEqual(calculated, exact, delta=0.01)
            for calculated, exact in zip(estimate.conditional_probabilities, reference):
                self.assertAlmostEqual(calculated, exact, delta=0.01)
            self.assertAlmostEqual(
                estimate.rectangle_joint,
                independent_union_probability(estimate.rectangle_probabilities),
                places=14,
            )
            self.assertAlmostEqual(
                estimate.conditional_joint,
                independent_union_probability(estimate.conditional_probabilities),
                places=14,
            )

    def test_zero_hits_do_not_fabricate_conditional_zero(self) -> None:
        gaussian = Gaussian2D((0.0, 0.0), 1.0)
        disk = Disk((0.0, 0.0), 0.1)
        estimate = shared_space_mc(
            (gaussian,),
            (disk,),
            n_samples=10,
            seed=3,
            bounds=Rectangle(-100.0, 100.0, -100.0, 100.0),
        ).disks[0]
        self.assertEqual(estimate.hit_count, 0)
        self.assertEqual(estimate.rectangle_probabilities, (0.0,))
        self.assertIsNone(estimate.conditional_probabilities)
        self.assertIsNone(estimate.conditional_joint)
        self.assertGreater(disk_reference(gaussian, disk), 0.0)



if __name__ == "__main__":
    unittest.main()
